"""
Mask R-CNN - Model Implementation

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Modified for DECIMER Segmentation 2024
- Removed unused imports
- Optimized inference path
- predict_on_batch() on numpy arrays instead of predict() (lower overhead), and not a direct
  eager call of the model, which leaks memory on every call under TF 2.15 (see detect())
"""

from __future__ import annotations

import os
import datetime
import re
import math
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

from . import utils

# Configure logging
logger = logging.getLogger(__name__)


def configure_gpu(gpu_id: Optional[int] = None, memory_growth: bool = True) -> None:
    """
    Configure GPU settings.

    Args:
        gpu_id: GPU device ID to use (None for all available)
        memory_growth: Whether to enable memory growth
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    gpus = tf.config.experimental.list_physical_devices("GPU")

    if memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.warning(f"Could not set memory growth: {e}")


# Apply default GPU configuration
configure_gpu(memory_growth=True)


# =============================================================================
# Utility Functions
# =============================================================================


class BatchNorm(KL.BatchNormalization):
    """Batch normalization layer with configurable training mode."""

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


def compute_backbone_shapes(config, image_shape: Tuple[int, ...]) -> np.ndarray:
    """Compute feature map shapes for each backbone stage."""
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    return np.array(
        [
            [
                int(math.ceil(image_shape[0] / stride)),
                int(math.ceil(image_shape[1] / stride)),
            ]
            for stride in config.BACKBONE_STRIDES
        ]
    )


# =============================================================================
# ResNet Backbone
# =============================================================================


def identity_block(
    input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True
):
    """Identity block for ResNet."""
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = f"res{stage}{block}_branch"
    bn_name_base = f"bn{stage}{block}_branch"

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + "2a", use_bias=use_bias)(
        input_tensor
    )
    x = BatchNorm(name=bn_name_base + "2a")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.Conv2D(
        nb_filter2,
        (kernel_size, kernel_size),
        padding="same",
        name=conv_name_base + "2b",
        use_bias=use_bias,
    )(x)
    x = BatchNorm(name=bn_name_base + "2b")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + "2c", use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + "2c")(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation("relu", name=f"res{stage}{block}_out")(x)
    return x


def conv_block(
    input_tensor,
    kernel_size,
    filters,
    stage,
    block,
    strides=(2, 2),
    use_bias=True,
    train_bn=True,
):
    """Convolutional block for ResNet."""
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = f"res{stage}{block}_branch"
    bn_name_base = f"bn{stage}{block}_branch"

    x = KL.Conv2D(
        nb_filter1,
        (1, 1),
        strides=strides,
        name=conv_name_base + "2a",
        use_bias=use_bias,
    )(input_tensor)
    x = BatchNorm(name=bn_name_base + "2a")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.Conv2D(
        nb_filter2,
        (kernel_size, kernel_size),
        padding="same",
        name=conv_name_base + "2b",
        use_bias=use_bias,
    )(x)
    x = BatchNorm(name=bn_name_base + "2b")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + "2c", use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + "2c")(x, training=train_bn)

    shortcut = KL.Conv2D(
        nb_filter3,
        (1, 1),
        strides=strides,
        name=conv_name_base + "1",
        use_bias=use_bias,
    )(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + "1")(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation("relu", name=f"res{stage}{block}_out")(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build ResNet graph."""
    assert architecture in ["resnet50", "resnet101"]

    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name="conv1", use_bias=True)(x)
    x = BatchNorm(name="bn_conv1")(x, training=train_bn)
    x = KL.Activation("relu")(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 2
    x = conv_block(
        x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1), train_bn=train_bn
    )
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b", train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block="c", train_bn=train_bn)

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a", train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="b", train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="c", train_bn=train_bn)
    C3 = x = identity_block(
        x, 3, [128, 128, 512], stage=3, block="d", train_bn=train_bn
    )

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a", train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(
            x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn
        )
    C4 = x

    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a", train_bn=train_bn)
        x = identity_block(
            x, 3, [512, 512, 2048], stage=5, block="b", train_bn=train_bn
        )
        C5 = x = identity_block(
            x, 3, [512, 512, 2048], stage=5, block="c", train_bn=train_bn
        )
    else:
        C5 = None

    return [C1, C2, C3, C4, C5]


# =============================================================================
# Proposal Layer
# =============================================================================


def apply_box_deltas_graph(boxes, deltas):
    """Apply box deltas in TensorFlow graph."""
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width

    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])

    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width

    return tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")


def clip_boxes_graph(boxes, window):
    """Clip boxes to window boundaries."""
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)

    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)

    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KL.Layer):
    """Generate proposal regions from RPN outputs."""

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.to_dict()
        config["proposal_count"] = self.proposal_count
        config["nms_threshold"] = self.nms_threshold
        return config

    def call(self, inputs):
        scores = inputs[0][:, :, 1]
        deltas = inputs[1] * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        anchors = inputs[2]

        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True).indices

        scores = utils.batch_slice(
            [scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU
        )
        deltas = utils.batch_slice(
            [deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU
        )
        pre_nms_anchors = utils.batch_slice(
            [anchors, ix],
            lambda a, x: tf.gather(a, x),
            self.config.IMAGES_PER_GPU,
            names=["pre_nms_anchors"],
        )

        boxes = utils.batch_slice(
            [pre_nms_anchors, deltas],
            lambda x, y: apply_box_deltas_graph(x, y),
            self.config.IMAGES_PER_GPU,
            names=["refined_anchors"],
        )

        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(
            boxes,
            lambda x: clip_boxes_graph(x, window),
            self.config.IMAGES_PER_GPU,
            names=["refined_anchors_clipped"],
        )

        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes,
                scores,
                self.proposal_count,
                self.nms_threshold,
                name="rpn_non_max_suppression",
            )
            proposals = tf.gather(boxes, indices)
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = utils.batch_slice([boxes, scores], nms, self.config.IMAGES_PER_GPU)

        if not tf.executing_eagerly():
            out_shape = self.compute_output_shape(None)
            proposals.set_shape(out_shape)

        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


# =============================================================================
# ROI Align Layer
# =============================================================================


def log2_graph(x):
    """Log base 2 in TensorFlow."""
    return tf.math.log(x) / tf.math.log(2.0)


class PyramidROIAlign(KL.Layer):
    """ROI pooling on feature pyramid."""

    def __init__(self, pool_shape, **kwargs):
        super().__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def get_config(self):
        config = super().get_config()
        config["pool_shape"] = self.pool_shape
        return config

    def call(self, inputs):
        boxes = inputs[0]
        image_meta = inputs[1]
        feature_maps = inputs[2:]

        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        image_shape = parse_image_meta_graph(image_meta)["image_shape"][0]
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)

        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(
            5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32))
        )
        roi_level = tf.squeeze(roi_level, 2)

        pooled = []
        box_to_level = []

        for i, level in enumerate(range(2, 6)):
            ix = tf.compat.v1.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)
            box_indices = tf.cast(ix[:, 0], tf.int32)

            box_to_level.append(ix)
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            pooled.append(
                tf.image.crop_and_resize(
                    feature_maps[i],
                    level_boxes,
                    box_indices,
                    self.pool_shape,
                    method="bilinear",
                )
            )

        pooled = tf.concat(pooled, axis=0)
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


# =============================================================================
# Detection Layer
# =============================================================================


def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine detections."""
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    deltas_specific = tf.gather_nd(deltas, indices)

    refined_rois = apply_box_deltas_graph(rois, deltas_specific * config.BBOX_STD_DEV)
    refined_rois = clip_boxes_graph(refined_rois, window)

    keep = tf.compat.v1.where(class_ids > 0)[:, 0]

    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.compat.v1.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[
            :, 0
        ]
        keep = tf.sets.intersection(
            tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0)
        )
        keep = tf.sparse.to_dense(keep)[0]

    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        ixs = tf.compat.v1.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD,
        )
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)], mode="CONSTANT", constant_values=-1)
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    nms_keep = tf.map_fn(
        nms_keep_map,
        unique_pre_nms_class_ids,
        fn_output_signature=tf.TensorSpec(shape=None, dtype=tf.int64),
    )
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.compat.v1.where(nms_keep > -1)[:, 0])

    keep = tf.sets.intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]

    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    detections = tf.concat(
        [
            tf.gather(refined_rois, keep),
            tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis],
        ],
        axis=1,
    )

    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], mode="CONSTANT")
    return detections


class DetectionLayer(KL.Layer):
    """Generate final detections."""

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.to_dict()
        return config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        m = parse_image_meta_graph(image_meta)
        image_shape = m["image_shape"][0]
        window = norm_boxes_graph(m["window"], image_shape[:2])

        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU,
        )

        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6],
        )

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


# =============================================================================
# RPN
# =============================================================================


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Build RPN graph."""
    shared = KL.Conv2D(
        512,
        (3, 3),
        padding="same",
        activation="relu",
        strides=anchor_stride,
        name="rpn_conv_shared",
    )(feature_map)

    x = KL.Conv2D(
        2 * anchors_per_location,
        (1, 1),
        padding="valid",
        activation="linear",
        name="rpn_class_raw",
    )(shared)
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
    rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    x = KL.Conv2D(
        anchors_per_location * 4,
        (1, 1),
        padding="valid",
        activation="linear",
        name="rpn_bbox_pred",
    )(shared)
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Build RPN Keras model."""
    input_feature_map = KL.Input(
        shape=[None, None, depth], name="input_rpn_feature_map"
    )
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


# =============================================================================
# FPN Heads
# =============================================================================


def fpn_classifier_graph(
    rois,
    feature_maps,
    image_meta,
    pool_size,
    num_classes,
    train_bn=True,
    fc_layers_size=1024,
):
    """Build FPN classifier head."""
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")(
        [rois, image_meta] + feature_maps
    )

    x = KL.TimeDistributed(
        KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
        name="mrcnn_class_conv1",
    )(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_class_bn1")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(
        x
    )
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_class_bn2")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)

    mrcnn_class_logits = KL.TimeDistributed(
        KL.Dense(num_classes), name="mrcnn_class_logits"
    )(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(
        mrcnn_class_logits
    )

    x = KL.TimeDistributed(
        KL.Dense(num_classes * 4, activation="linear"), name="mrcnn_bbox_fc"
    )(shared)

    s = K.int_shape(x)
    if s[1] is None:
        mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
    else:
        mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(
    rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True
):
    """Build FPN mask head."""
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")(
        [rois, image_meta] + feature_maps
    )

    for i in range(4):
        x = KL.TimeDistributed(
            KL.Conv2D(256, (3, 3), padding="same"), name=f"mrcnn_mask_conv{i+1}"
        )(x)
        x = KL.TimeDistributed(BatchNorm(), name=f"mrcnn_mask_bn{i+1}")(
            x, training=train_bn
        )
        x = KL.Activation("relu")(x)

    x = KL.TimeDistributed(
        KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
        name="mrcnn_mask_deconv",
    )(x)
    x = KL.TimeDistributed(
        KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
        name="mrcnn_mask",
    )(x)
    return x


# =============================================================================
# Loss Functions (kept for training compatibility)
# =============================================================================


def smooth_l1_loss(y_true, y_pred):
    """Smooth L1 loss."""
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    return (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN class loss."""
    rpn_match = tf.squeeze(rpn_match, -1)
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    indices = tf.compat.v1.where(K.not_equal(rpn_match, 0))
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)

    loss = K.sparse_categorical_crossentropy(
        target=anchor_class, output=rpn_class_logits, from_logits=True
    )
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """RPN bbox loss."""
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.compat.v1.where(K.equal(rpn_match, 1))
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    """MRCNN class loss."""
    target_class_ids = tf.cast(target_class_ids, "int64")
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits
    )
    loss = loss * pred_active
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """MRCNN bbox loss."""
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    positive_roi_ix = tf.compat.v1.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64
    )
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    loss = K.switch(
        tf.size(target_bbox) > 0,
        smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
        tf.constant(0.0),
    )
    return K.mean(loss)


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """MRCNN mask loss."""
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(
        pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4])
    )
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    positive_ix = tf.compat.v1.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    loss = K.switch(
        tf.size(y_true) > 0,
        K.binary_crossentropy(target=y_true, output=y_pred),
        tf.constant(0.0),
    )
    return K.mean(loss)


# =============================================================================
# Data Formatting
# =============================================================================


def compose_image_meta(
    image_id, original_image_shape, image_shape, window, scale, active_class_ids
):
    """Compose image metadata array."""
    meta = np.array(
        [image_id]
        + list(original_image_shape)
        + list(image_shape)
        + list(window)
        + [scale]
        + list(active_class_ids)
    )
    return meta


def parse_image_meta_graph(meta):
    """Parse image metadata tensor."""
    return {
        "image_id": meta[:, 0],
        "original_image_shape": meta[:, 1:4],
        "image_shape": meta[:, 4:7],
        "window": meta[:, 7:11],
        "scale": meta[:, 11],
        "active_class_ids": meta[:, 12:],
    }


def mold_image(images, config):
    """Normalize image for model input."""
    return images.astype(np.float32) - config.MEAN_PIXEL


# =============================================================================
# Graph Utilities
# =============================================================================


def batch_pack_graph(x, counts, num_rows):
    """Pack batch slices."""
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, : counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Normalize boxes to 0-1 range."""
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0.0, 0.0, 1.0, 1.0])
    return tf.divide(boxes - shift, scale)


# =============================================================================
# MaskRCNN Model Class
# =============================================================================


class MaskRCNN:
    """Mask R-CNN model for inference."""

    def __init__(self, mode: str, config, model_dir: str):
        """
        Initialize model.

        Args:
            mode: "training" or "inference"
            config: Configuration object
            model_dir: Directory for logs/checkpoints
        """
        assert mode in ["training", "inference"]
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode: str, config):
        """Build the Mask R-CNN architecture."""
        assert mode in ["training", "inference"]

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise ValueError(f"Image size must be divisible by 64. Got {h}x{w}")

        # Inputs
        input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image"
        )
        input_image_meta = KL.Input(
            shape=[config.IMAGE_META_SIZE], name="input_image_meta"
        )

        if mode == "training":
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32
            )
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32
            )
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32
            )
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32
            )
            gt_boxes = KL.Lambda(
                lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3])
            )(input_gt_boxes)

            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks",
                    dtype=bool,
                )
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks",
                    dtype=bool,
                )
        else:
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Backbone
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(
                input_image, stage5=True, train_bn=config.TRAIN_BN
            )
        else:
            _, C2, C3, C4, C5 = resnet_graph(
                input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN
            )

        # FPN
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c5p5")(C5)
        P4 = KL.Add(name="fpn_p4add")(
            [
                KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c4p4")(C4),
            ]
        )
        P3 = KL.Add(name="fpn_p3add")(
            [
                KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c3p3")(C3),
            ]
        )
        P2 = KL.Add(name="fpn_p2add")(
            [
                KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c2p2")(C2),
            ]
        )

        P2 = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2"
        )(P2)
        P3 = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3"
        )(P3)
        P4 = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4"
        )(P4)
        P5 = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5"
        )(P5)
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)

            class ConstLayer(tf.keras.layers.Layer):
                def __init__(self, x, name=None):
                    super().__init__(name=name)
                    self.x = tf.Variable(x)

                def call(self, inputs):
                    return self.x

            anchors = ConstLayer(anchors, name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN
        rpn = build_rpn_model(
            config.RPN_ANCHOR_STRIDE,
            len(config.RPN_ANCHOR_RATIOS),
            config.TOP_DOWN_PYRAMID_SIZE,
        )

        layer_outputs = [rpn([p]) for p in rpn_feature_maps]

        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [
            KL.Concatenate(axis=1, name=n)(list(o))
            for o, n in zip(outputs, output_names)
        ]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Proposals
        proposal_count = (
            config.POST_NMS_ROIS_TRAINING
            if mode == "training"
            else config.POST_NMS_ROIS_INFERENCE
        )
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config,
        )([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            from .detection_target_layer import DetectionTargetLayer

            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"]
            )(input_image_meta)

            rois, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
                config, name="proposal_targets"
            )([rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
                rois,
                mrcnn_feature_maps,
                input_image_meta,
                config.POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE,
            )

            mrcnn_mask = build_fpn_mask_graph(
                rois,
                mrcnn_feature_maps,
                input_image_meta,
                config.MASK_POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
            )

            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            rpn_class_loss = KL.Lambda(
                lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss"
            )([input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(
                lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss"
            )([input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(
                lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss"
            )([target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(
                lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss"
            )([target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(
                lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss"
            )([target_mask, target_class_ids, mrcnn_mask])

            inputs = [
                input_image,
                input_image_meta,
                input_rpn_match,
                input_rpn_bbox,
                input_gt_class_ids,
                input_gt_boxes,
                input_gt_masks,
            ]
            outputs = [
                rpn_class_logits,
                rpn_class,
                rpn_bbox,
                mrcnn_class_logits,
                mrcnn_class,
                mrcnn_bbox,
                mrcnn_mask,
                rpn_rois,
                output_rois,
                rpn_class_loss,
                rpn_bbox_loss,
                class_loss,
                bbox_loss,
                mask_loss,
            ]
            model = KM.Model(inputs, outputs, name="mask_rcnn")
        else:
            # Inference mode
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
                rpn_rois,
                mrcnn_feature_maps,
                input_image_meta,
                config.POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE,
            )

            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta]
            )

            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(
                detection_boxes,
                mrcnn_feature_maps,
                input_image_meta,
                config.MASK_POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
            )

            model = KM.Model(
                [input_image, input_image_meta, input_anchors],
                [
                    detections,
                    mrcnn_class,
                    mrcnn_bbox,
                    mrcnn_mask,
                    rpn_rois,
                    rpn_class,
                    rpn_bbox,
                ],
                name="mask_rcnn",
            )

        return model

    def load_weights(
        self, filepath: str, by_name: bool = False, exclude: Optional[List[str]] = None
    ) -> None:
        """Load model weights."""
        import h5py
        from tensorflow.python.keras.saving import hdf5_format

        if exclude:
            by_name = True

        with h5py.File(filepath, mode="r") as f:
            if "layer_names" not in f.attrs and "model_weights" in f:
                f = f["model_weights"]

            keras_model = self.keras_model
            layers = (
                keras_model.inner_model.layers
                if hasattr(keras_model, "inner_model")
                else keras_model.layers
            )

            if exclude:
                layers = [layer for layer in layers if layer.name not in exclude]

            if by_name:
                hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
            else:
                hdf5_format.load_weights_from_hdf5_group(f, layers)

        self.set_log_dir(filepath)

    def set_log_dir(self, model_path: Optional[str] = None) -> None:
        """Set log directory."""
        self.epoch = 0
        now = datetime.datetime.now()

        if model_path:
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, str(model_path))
            if m:
                now = datetime.datetime(
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3)),
                    int(m.group(4)),
                    int(m.group(5)),
                )
                self.epoch = int(m.group(6)) - 1 + 1

        self.log_dir = os.path.join(
            self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now)
        )
        self.checkpoint_path = os.path.join(
            self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(self.config.NAME.lower())
        ).replace("*epoch*", "{epoch:04d}")

    def get_anchors(self, image_shape: Tuple[int, ...]) -> np.ndarray:
        """Generate anchors for given image shape (cached)."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)

        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}

        cache_key = tuple(image_shape)
        if cache_key not in self._anchor_cache:
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE,
            )
            self.anchors = a
            self._anchor_cache[cache_key] = utils.norm_boxes(a, image_shape[:2])

        return self._anchor_cache[cache_key]

    def mold_inputs(
        self, images: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare images for model input."""
        molded_images = []
        image_metas = []
        windows = []

        for image in images:
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE,
            )
            molded_image = mold_image(molded_image, self.config)

            image_meta = compose_image_meta(
                0,
                image.shape,
                molded_image.shape,
                window,
                scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32),
            )

            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)

        return np.stack(molded_images), np.stack(image_metas), np.stack(windows)

    def unmold_detections(
        self, detections, mrcnn_mask, original_image_shape, image_shape, window
    ):
        """Convert detections to final format."""
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh, ww = wy2 - wy1, wx2 - wx1
        scale = np.array([wh, ww, wh, ww])

        boxes = np.divide(boxes - shift, scale)
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0
        )[0]

        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        full_masks = []
        for i in range(N):
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)

        full_masks = (
            np.stack(full_masks, axis=-1)
            if full_masks
            else np.empty(original_image_shape[:2] + (0,))
        )

        return boxes, class_ids, scores, full_masks

    def detect(
        self, images: List[np.ndarray], verbose: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Run detection on images.

        OPTIMIZATION: runs the forward pass with predict_on_batch() on numpy arrays, which has
        less overhead than predict() (no data pipeline, no callbacks). Do not replace it with a
        direct eager call of the model, self.keras_model(tensors, training=False): under TF 2.15
        on CPU that leaks about 3.5 MB per call (200 calls on 3 images grew the process from
        1.79 to 2.49 GB, linearly). predict_on_batch() stays flat (1.77 to 1.73 GB over 200
        calls), is about 30 % faster, and returns the same detections.
        """
        assert self.mode == "inference", "Model must be in inference mode"
        assert len(images) == self.config.BATCH_SIZE

        molded_images, image_metas, windows = self.mold_inputs(images)

        image_shape = molded_images[0].shape
        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        # float32 numpy arrays in, as the tf.constant(..., dtype=tf.float32) inputs were before
        inputs = [
            np.asarray(molded_images, dtype=np.float32),
            np.asarray(image_metas, dtype=np.float32),
            np.asarray(anchors, dtype=np.float32),
        ]

        # numpy arrays out, in the order of the model definition
        outputs = self.keras_model.predict_on_batch(inputs)

        detections = outputs[0]
        mrcnn_mask = outputs[3]

        results = []
        for i, image in enumerate(images):
            (
                final_rois,
                final_class_ids,
                final_scores,
                final_masks,
            ) = self.unmold_detections(
                detections[i],
                mrcnn_mask[i],
                image.shape,
                molded_images[i].shape,
                windows[i],
            )
            results.append(
                {
                    "rois": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                    "masks": final_masks,
                }
            )

        return results
