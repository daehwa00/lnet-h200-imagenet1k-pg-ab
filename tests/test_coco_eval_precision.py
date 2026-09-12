from contextlib import contextmanager

import pytest
import torch

from dense_transfer import data, metrics


@pytest.mark.parametrize('device,bf16,expected', [('cuda', True, True), ('cuda', False, False), ('cpu', True, False)])
def test_coco_eval_obeys_precision_and_restores_training(monkeypatch, device, bf16, expected):
    active = [False]
    @contextmanager
    def autocast(*, device_type, dtype):
        assert device_type == 'cuda' and dtype == torch.bfloat16
        active[0] = True
        try:
            yield
        finally:
            active[0] = False
    class Model(torch.nn.Module):
        def forward(self, images):
            assert not self.training and torch.is_inference_mode_enabled()
            assert active[0] is expected
            return [{'boxes': torch.empty(0, 4)}]
    class Evaluator:
        def __init__(self, dataset):
            pass
        def update(self, predictions, image_ids):
            assert image_ids == [1]
        def compute(self):
            return {'bbox_AP': 0.0}
    monkeypatch.setattr(torch, 'autocast', autocast)
    monkeypatch.setattr(data, '_move_nested', lambda value, device: value)
    monkeypatch.setattr(metrics, 'COCOEvaluator', Evaluator)
    model = Model().train()
    assert data.evaluate(model, [([torch.zeros(3, 8, 8)], [{'image_id': 1}])], device, 'coco', bf16) == {'bbox_AP': 0.0}
    assert model.training and not active[0]


def test_encoded_accumulator_matches_existing_metrics(tmp_path):
    from test_dense_transfer_data import _write_coco_fixture
    _write_coco_fixture(tmp_path)
    dataset = data.build_dataset('coco', tmp_path, 'train', train=False)
    _, target = dataset[0]
    pred = {'boxes': target['boxes'], 'labels': target['labels'],
            'scores': torch.ones(len(target['boxes'])), 'masks': target['masks'][:, None].float()}
    meter = metrics.COCOEvaluator(dataset)
    meter.update([pred], image_ids=[1])
    assert meter.compute() == metrics.evaluate_coco(dataset, [{**pred, 'image_id': 1}])
    assert isinstance(meter.rows['segm'][0]['segmentation']['counts'], (str, bytes))
    meter.reset()
    assert meter.rows == {'bbox': [], 'segm': []}
