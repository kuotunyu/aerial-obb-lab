from __future__ import annotations

import numpy as np

from obbkit.viz import _imread_unicode, _imwrite_unicode


def test_image_round_trip_under_unicode_path(tmp_path) -> None:
    folder = tmp_path / "中文 路徑"
    folder.mkdir()
    path = folder / "測試影像.jpg"
    image = np.full((24, 32, 3), 127, dtype=np.uint8)

    _imwrite_unicode(path, image)
    decoded = _imread_unicode(path)

    assert decoded is not None
    assert decoded.shape == image.shape
