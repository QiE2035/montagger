"""Preprocessing: canvas layout, channel order, normalization values."""

import io

import numpy as np
from PIL import Image

from montagger.engine import preprocess
from conftest import make_profile


def solid_png(w, h, color, alpha=None):
    img = Image.new("RGBA", (w, h), color + (alpha,) if alpha is not None else color + (255,))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_decode_applies_size_and_returns_rgba():
    rgba = preprocess.decode(solid_png(6, 4, (10, 20, 30)))
    assert (rgba.width, rgba.height) == (6, 4)


def test_not_an_image():
    import pytest

    with pytest.raises(preprocess.NotAnImage):
        preprocess.decode(b"not an image at all")


def test_white_pad_wide_image():
    profile = make_profile()
    canvas = preprocess.pad_and_resize(preprocess.decode(solid_png(8, 4, (10, 20, 30))), 4, profile)
    assert canvas.shape == (4, 4, 3)
    # 8x4 into 4x4: scale to 4x2, centered vertically, white bands top+bottom.
    assert (canvas[0] == 255).all()
    assert (canvas[3] == 255).all()
    assert (canvas[1:3] == np.array([10, 20, 30], dtype=np.uint8)).all()


def test_mean_color_pad():
    profile = make_profile(pad="mean_color_aspect")  # fill_color zero -> camie default
    canvas = preprocess.pad_and_resize(preprocess.decode(solid_png(4, 8, (1, 2, 3))), 4, profile)
    # 4x8 into 4x4: scale to 2x4, centered horizontally (x1..x2), camie fill bands at x0/x3.
    assert (canvas[0, 0] == preprocess._CAMIE_FILL).all()
    assert (canvas[2, 1] == np.array([1, 2, 3], dtype=np.uint8)).all()
    assert (canvas[3, 3] == preprocess._CAMIE_FILL).all()


def test_nhwc_bgr_order():
    profile = make_profile(layout="nhwc", channels="bgr", normalize="none")
    tensor = preprocess.build_tensor(np.full((4, 4, 3), [1, 2, 3], dtype=np.uint8), 4, profile)
    assert tensor.shape == (1, 4, 4, 3)
    assert tensor[0, 0, 0, 0] == 3 and tensor[0, 0, 0, 2] == 1  # bgr swap
    assert tensor.dtype == np.float32


def test_nchw_clip_normalization():
    import pytest

    profile = make_profile(layout="nchw", channels="rgb", normalize="clip")
    canvas = np.zeros((2, 2, 3), dtype=np.uint8)
    tensor = preprocess.build_tensor(canvas, 2, profile)
    assert tensor.shape == (1, 3, 2, 2)
    expected = (0.0 - preprocess._CLIP_MEAN[0]) / preprocess._CLIP_STD[0]
    assert tensor[0, 0, 0, 0] == pytest.approx(expected, rel=1e-5)


def test_premultiplication_for_transparency():
    profile = make_profile()
    rgba = preprocess.decode(solid_png(1, 1, (200, 100, 50), alpha=128))
    canvas = preprocess.pad_and_resize(rgba, 1, profile)
    # Premultiplied against black, rounded: 200*128/255 ~= 100
    assert canvas[0, 0, 0] == 100


def test_infer_input_size():
    assert preprocess.infer_input_size([1, 448, 448, 3], "nhwc") == 448
    assert preprocess.infer_input_size(["n", 3, 384, 384], "nchw") == 384
    assert preprocess.infer_input_size(["n", 3, "h", 3], "nchw") == 0
    assert preprocess.infer_input_size([1, 3], "nchw") == 0
