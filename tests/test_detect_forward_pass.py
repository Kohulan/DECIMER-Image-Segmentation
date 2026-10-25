"""MaskRCNN.detect() runs the forward pass with predict_on_batch(), not as a direct eager call.

A direct call, `self.keras_model(tensors, training=False)`, leaks about 3.5 MB per call under TF 2.15
on CPU (200 calls on 3 images grew the process from 1.79 to 2.49 GB, linearly; a service replica was
OOM-killed after a day of health checks). predict_on_batch() on numpy arrays stayed flat, was about
30 % faster, and gave the same detections.

Most of these tests need no weights: the Keras model is a recorder that fails if it is called
directly. The last two run the real model, if its weights are in the PYSTOW_HOME directory: one
checks what predict_on_batch() takes and returns, the other that its outputs are those of the old
direct call on the same inputs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from decimer_segmentation.mrcnn.model import MaskRCNN

HEIGHT = WIDTH = 128  # size of the molded image the fake model "sees"
N_DETECTIONS = 3


class FakeKerasModel:
    """Records predict_on_batch(); any other way of running the model fails the test."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.batches = []

    def predict_on_batch(self, inputs):
        self.batches.append(inputs)
        return self.outputs

    def __call__(self, *args, **kwargs):
        raise AssertionError(
            "direct eager call of the model: it leaks memory on every call"
        )

    def predict(self, *args, **kwargs):
        raise AssertionError("predict() has the data pipeline and callbacks overhead")


def model_outputs(batch_size):
    """Seven numpy arrays in the order of the model definition (detections at 0, masks at 3)."""
    rng = np.random.default_rng(0)
    return [
        rng.random((batch_size, N_DETECTIONS, 6), dtype=np.float32),  # detections
        rng.random((batch_size, 5, 2), dtype=np.float32),  # mrcnn_class
        rng.random((batch_size, 5, 2, 4), dtype=np.float32),  # mrcnn_bbox
        rng.random((batch_size, N_DETECTIONS, 28, 28, 2), dtype=np.float32),  # masks
        rng.random((batch_size, 5, 4), dtype=np.float32),  # rpn_rois
        rng.random((batch_size, 7, 2), dtype=np.float32),  # rpn_class
        rng.random((batch_size, 7, 4), dtype=np.float32),  # rpn_bbox
    ]


def make_detector(keras_model, batch_size=1):
    """A MaskRCNN without a built network: everything detect() calls besides the model is stubbed."""
    detector = MaskRCNN.__new__(MaskRCNN)
    detector.mode = "inference"
    detector.config = SimpleNamespace(BATCH_SIZE=batch_size)
    detector.keras_model = keras_model
    detector.molded = {
        # float64 metas and anchors: the old tf.constant(..., dtype=tf.float32) cast them
        "images": np.full((batch_size, HEIGHT, WIDTH, 3), 1.5, dtype=np.float32),
        "metas": np.full((batch_size, 14), 2.0, dtype=np.float64),
        "anchors": np.full((4, 4), 3.0, dtype=np.float64),
    }
    detector.mold_inputs = lambda images: (
        detector.molded["images"],
        detector.molded["metas"],
        np.tile(np.array([0, 0, HEIGHT, WIDTH]), (batch_size, 1)),
    )
    detector.get_anchors = lambda image_shape: detector.molded["anchors"]
    detector.unmolded = []

    def unmold(detections, mrcnn_mask, original_image_shape, image_shape, window):
        detector.unmolded.append((detections, mrcnn_mask, original_image_shape))
        # the rois say which image's results these are
        return (
            np.full((1, 4), len(detector.unmolded)),
            np.array([1]),
            np.array([0.5]),
            np.zeros(original_image_shape[:2] + (1,), dtype=bool),
        )

    detector.unmold_detections = unmold
    return detector


@pytest.mark.parametrize("batch_size", [1, 2])
def test_detect_runs_the_forward_pass_with_predict_on_batch(batch_size):
    fake = FakeKerasModel(model_outputs(batch_size))
    detector = make_detector(fake, batch_size)
    images = [np.full((60 + i, 80, 3), 255, dtype=np.uint8) for i in range(batch_size)]

    results = detector.detect(images)

    assert len(fake.batches) == 1  # once, for the whole batch
    (inputs,) = fake.batches
    assert len(inputs) == 3
    # numpy float32 in (a tf.constant has no place here), in the order of the model's inputs:
    # molded images, image metas, anchors broadcast over the batch
    assert all(type(a) is np.ndarray and a.dtype == np.float32 for a in inputs)
    np.testing.assert_array_equal(inputs[0], detector.molded["images"])
    np.testing.assert_array_equal(
        inputs[1], detector.molded["metas"].astype(np.float32)
    )
    assert inputs[2].shape == (batch_size, 4, 4)
    assert np.all(inputs[2] == 3.0)

    # every image gets its own row of the model's detections (output 0) and masks (output 3),
    # unchanged, and the results come back in the order of the images
    assert len(results) == batch_size
    for i, image in enumerate(images):
        detections, mrcnn_mask, original_image_shape = detector.unmolded[i]
        np.testing.assert_array_equal(detections, fake.outputs[0][i])
        np.testing.assert_array_equal(mrcnn_mask, fake.outputs[3][i])
        assert original_image_shape == image.shape
        assert set(results[i]) == {"rois", "class_ids", "scores", "masks"}
        assert results[i]["rois"][0, 0] == i + 1
        assert results[i]["masks"].shape == image.shape[:2] + (1,)


def test_detect_does_not_need_tensors_back():
    """predict_on_batch() returns numpy arrays, which have no .numpy(): detect() must not use it."""
    outputs = model_outputs(1)
    assert not any(hasattr(o, "numpy") for o in outputs)
    detector = make_detector(FakeKerasModel(outputs))

    detector.detect([np.full((60, 80, 3), 255, dtype=np.uint8)])


def test_detect_returns_what_unmold_detections_makes_of_the_outputs():
    """With the real unmold_detections, a box at 25 % to 75 % of a 100 x 100 page comes back."""
    outputs = model_outputs(1)
    outputs[0] = np.zeros((1, N_DETECTIONS, 6), dtype=np.float32)
    # y1, x1, y2, x2, class id, score
    outputs[0][0, 0] = [0.25, 0.25, 0.75, 0.75, 1, 0.9]
    outputs[3] = np.ones((1, N_DETECTIONS, 28, 28, 2), dtype=np.float32)
    detector = make_detector(FakeKerasModel(outputs))
    del detector.unmold_detections  # back to the class's own method

    (result,) = detector.detect([np.full((100, 100, 3), 255, dtype=np.uint8)])

    np.testing.assert_array_equal(result["rois"], [[25, 25, 75, 75]])
    np.testing.assert_array_equal(result["class_ids"], [1])
    np.testing.assert_allclose(result["scores"], [0.9], rtol=1e-6)
    assert result["masks"].shape == (100, 100, 1)
    assert result["masks"][50, 50, 0]
    assert not result["masks"][5, 5, 0]


REPO_ROOT = Path(__file__).resolve().parents[1]


def real_model():
    """The loaded model, or a skip if its weights are not in PYSTOW_HOME (never a download)."""
    import pystow

    from decimer_segmentation import decimer_segmentation as ds

    weights = pystow.join(
        "DECIMER-Segmentation_model", name="mask_rcnn_molecule.h5", ensure_exists=False
    )
    if not weights.is_file():
        pytest.skip(f"no weights at {weights}")
    return ds.get_model()


def caffeine_page():
    """A page with two caffeine structures (tracked in the repo), transparency on white, BGR."""
    rgba = cv2.imread(
        str(REPO_ROOT / "Validation" / "Abstract1.png"), cv2.IMREAD_UNCHANGED
    )
    alpha = rgba[:, :, 3:] / 255.0
    return np.round(rgba[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)


def detect_recording_predict_on_batch(model, monkeypatch, image):
    """detect() on the real model, with what predict_on_batch() was given and returned."""
    seen = []
    real_predict_on_batch = model.keras_model.predict_on_batch

    def spy(inputs):
        outputs = real_predict_on_batch(inputs)
        seen.append((inputs, outputs))
        return outputs

    monkeypatch.setattr(model.keras_model, "predict_on_batch", spy)
    (result,) = model.detect([image])
    ((inputs, outputs),) = seen
    return inputs, outputs, result


@pytest.mark.slow
def test_real_model_returns_numpy_outputs_from_predict_on_batch(monkeypatch):
    """The real Keras model takes the numpy inputs and returns numpy arrays, detections first."""
    image = caffeine_page()

    inputs, outputs, result = detect_recording_predict_on_batch(
        real_model(), monkeypatch, image
    )

    assert all(type(a) is np.ndarray for a in inputs)
    assert all(type(o) is np.ndarray for o in outputs)
    assert len(outputs) == 7
    assert outputs[0].shape[-1] == 6  # detections: y1, x1, y2, x2, class id, score
    assert outputs[3].shape[-1] == 2  # masks of both classes
    assert set(result) == {"rois", "class_ids", "scores", "masks"}
    # the two caffeine structures at least (the model also boxes the drawing in the middle)
    assert len(result["rois"]) >= 2
    assert result["masks"].shape == image.shape[:2] + (len(result["rois"]),)


@pytest.mark.slow
def test_predict_on_batch_gives_the_same_outputs_as_the_old_direct_call(monkeypatch):
    """The call detect() used to make, on the very inputs it now gives predict_on_batch()."""
    import tensorflow as tf

    model = real_model()
    inputs, outputs, _ = detect_recording_predict_on_batch(
        model, monkeypatch, caffeine_page()
    )
    # real detections, not just the padding of the output
    assert (outputs[0][0, :, 4] > 0).sum() >= 2

    # once, because a direct eager call leaks about 3.5 MB each time
    direct = model.keras_model(
        [tf.constant(a, dtype=tf.float32) for a in inputs], training=False
    )

    # The two outputs detect() reads. Graph and eager kernels differ by float32 noise: here up to
    # 1.5e-6 in the detections and 1.1e-5 in the mask probabilities (after the 0.5 threshold of
    # unmold_mask no pixel differs); the tolerances are about ten times that.
    for index, atol in ((0, 1e-5), (3, 1e-4)):
        np.testing.assert_allclose(
            outputs[index], direct[index].numpy(), rtol=0, atol=atol
        )
