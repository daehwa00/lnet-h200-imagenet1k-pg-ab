"""Only the externally verified completed v3 K96seed501 may be carried forward."""
import json
from pathlib import Path
from in1k10_subset import SHA256

CAMPAIGN='simclr10-networkfix-v4'
FIXED_PROFILE={'ipc':'memfd','batch_size':512,'workers':8,'prefetch_factor':1}


def load_completed(path=None):
    path=Path(path) if path else Path(__file__).resolve().parents[1]/'h200/in1k10_completed_v3.json'
    receipt=json.loads(path.read_text())
    if receipt.get('subset_sha256')!=SHA256:raise ValueError('Completed result dataset mismatch')
    recipe=receipt.get('recipe',{})
    expected={'epochs':100,'physical_batch':512,'effective_batch':512,'workers':8,'image_size':224,
        'optimizer':'AdamW','learning_rate':.003,'weight_decay':.05,'warmup_epochs':5,'evaluation_policy':'final_epoch_only'}
    if recipe!=expected:raise ValueError('Completed result recipe mismatch')
    rows=receipt.get('completed',[])
    if len(rows)!=1:raise ValueError('Only one completed prior run is verified')
    row=rows[0]
    required={'job':'va_k96-501','model':'va_k96','seed':501,'completed_epochs':100,'global_step':25000,
        'validation_examples':50000,'top1':.55904,'top5':.79816,'wandb_state_verified':'finished',
        'wandb_run':'abe62b007c5346a1','source_code':'eccc623cbfcb3ff886de980c8c3dfedd4e2f4c07'}
    if any(row.get(k)!=v for k,v in required.items()):raise ValueError('Incomplete or unverified prior result')
    return rows
