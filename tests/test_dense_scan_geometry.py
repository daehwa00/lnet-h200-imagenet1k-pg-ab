from lnet.pac_triton_product_scan_coarse4 import dense_scan_geometry


def test_classification_shapes_keep_existing_tuning():
    for height in (7, 14, 28, 56, 64):
        assert dense_scan_geometry(height) is None


def test_dense_scan_tiles_bound_independent_mode_work():
    for height, modes in ((128, 4), (200, 2), (256, 2), (336, 1), (512, 1)):
        geometry = dense_scan_geometry(height)
        assert geometry.blocks == {"BLOCK_LINES": 1, "BLOCK_MODES": modes}
        assert geometry.num_warps == 4
        assert geometry.num_stages == 1
        assert (1 << (height - 1).bit_length()) * modes <= 512
