from __future__ import annotations

import numpy as np

from obbkit.analysis import ObbObject, _hbb_iou_matrix, inflation_stats


def test_diamond_has_expected_area_inflation() -> None:
    diamond = ObbObject(
        image="sample",
        cls=0,
        poly=np.array([[1.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 1.0]]),
    )

    assert diamond.obb_area == 2.0
    assert diamond.hbb_area == 4.0
    assert inflation_stats([diamond])["plane"]["mean"] == 2.0


def test_hbb_iou_matrix_is_symmetric_and_bounded() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 1.0, 3.0, 3.0],
            [5.0, 5.0, 6.0, 6.0],
        ]
    )

    iou = _hbb_iou_matrix(boxes)

    np.testing.assert_allclose(iou, iou.T)
    np.testing.assert_allclose(np.diag(iou), np.ones(3))
    assert iou[0, 1] == 1.0 / 7.0
    assert iou[0, 2] == 0.0
