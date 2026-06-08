"""
Define the segmentation losses and patch-level metrics used during training.
The clean physical baseline uses these helpers to favor missed-shell recall over
strict pixel precision.
"""

import tensorflow as tf
from tensorflow import keras


def _loss_cfg(cfg):
    """Read the loss block while supporting older optimizer-only configs."""
    return cfg.get("loss") or {"name": cfg.get("optim", {}).get("loss", "binary_crossentropy")}


def weighted_bce(y_true, y_pred, positive_weight: float = 1.0):
    """Compute binary cross-entropy with extra weight on catalog shell pixels."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
    bce = -(positive_weight * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce)


def tversky_index(y_true, y_pred, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
    """Compute the soft Tversky overlap used to bias training toward recall."""
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def tversky_loss(y_true, y_pred, alpha: float = 0.3, beta: float = 0.7):
    """Convert Tversky overlap into a minimization loss."""
    return 1.0 - tversky_index(y_true, y_pred, alpha=alpha, beta=beta)


@keras.utils.register_keras_serializable(package="pv_shells")
class WeightedBCELoss(keras.losses.Loss):
    """Keras-serializable wrapper for weighted binary cross-entropy."""

    def __init__(self, positive_weight: float = 1.0, name: str = "weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.positive_weight = float(positive_weight)

    def call(self, y_true, y_pred):
        """Evaluate the weighted BCE term for one batch."""
        return weighted_bce(y_true, y_pred, positive_weight=self.positive_weight)

    def get_config(self):
        """Store custom loss settings inside saved Keras models."""
        cfg = super().get_config()
        cfg.update({"positive_weight": self.positive_weight})
        return cfg


@keras.utils.register_keras_serializable(package="pv_shells")
class TverskyLoss(keras.losses.Loss):
    """Keras-serializable recall-friendly Tversky loss."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, name: str = "tversky", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def call(self, y_true, y_pred):
        """Evaluate Tversky loss for one batch."""
        return tversky_loss(y_true, y_pred, alpha=self.alpha, beta=self.beta)

    def get_config(self):
        """Store Tversky alpha and beta in saved Keras models."""
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "beta": self.beta})
        return cfg


@keras.utils.register_keras_serializable(package="pv_shells")
class BCETverskyLoss(keras.losses.Loss):
    """Blend BCE stability with Tversky's stronger false-negative penalty."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        bce_weight: float = 0.5,
        tversky_weight: float = 0.5,
        positive_weight: float = 1.0,
        weighted_bce: bool = False,
        name: str = "bce_tversky",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.bce_weight = float(bce_weight)
        self.tversky_weight = float(tversky_weight)
        self.positive_weight = float(positive_weight)
        self.weighted_bce = bool(weighted_bce)
        self._bce = keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        """Evaluate the blended BCE plus Tversky objective for one batch."""
        if self.weighted_bce:
            bce_part = weighted_bce(y_true, y_pred, positive_weight=self.positive_weight)
        else:
            bce_part = self._bce(y_true, y_pred)
        tv_part = tversky_loss(y_true, y_pred, alpha=self.alpha, beta=self.beta)
        return self.bce_weight * bce_part + self.tversky_weight * tv_part

    def get_config(self):
        """Store all blend weights and Tversky settings in saved models."""
        cfg = super().get_config()
        cfg.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "bce_weight": self.bce_weight,
                "tversky_weight": self.tversky_weight,
                "positive_weight": self.positive_weight,
                "weighted_bce": self.weighted_bce,
            }
        )
        return cfg


@keras.utils.register_keras_serializable(package="pv_shells")
class PatchPrecision(keras.metrics.Metric):
    """Track whether predicted probability exists anywhere in a patch."""

    def __init__(self, threshold: float = 0.075, name: str | None = None, **kwargs):
        tag = str(threshold).replace(".", "p")
        super().__init__(name=name or f"patch_precision_{tag}", **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate patch-level true positives and false positives."""
        y_true_patch = tf.reduce_any(tf.cast(y_true > 0.5, tf.bool), axis=[1, 2, 3])
        y_pred_patch = tf.reduce_max(tf.cast(y_pred, tf.float32), axis=[1, 2, 3]) >= self.threshold
        self.tp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_patch, y_pred_patch), tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true_patch), y_pred_patch), tf.float32)))

    def result(self):
        """Return precision over all patches seen by this metric instance."""
        return self.tp / tf.maximum(1.0, self.tp + self.fp)

    def reset_state(self):
        """Clear accumulated patch counts between epochs."""
        self.tp.assign(0.0)
        self.fp.assign(0.0)

    def get_config(self):
        """Store the detection threshold in saved Keras metric config."""
        cfg = super().get_config()
        cfg.update({"threshold": self.threshold})
        return cfg


@keras.utils.register_keras_serializable(package="pv_shells")
class PatchRecall(keras.metrics.Metric):
    """Track whether each labeled patch receives at least one predicted pixel."""

    def __init__(self, threshold: float = 0.075, name: str | None = None, **kwargs):
        tag = str(threshold).replace(".", "p")
        super().__init__(name=name or f"patch_recall_{tag}", **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate patch-level true positives and false negatives."""
        y_true_patch = tf.reduce_any(tf.cast(y_true > 0.5, tf.bool), axis=[1, 2, 3])
        y_pred_patch = tf.reduce_max(tf.cast(y_pred, tf.float32), axis=[1, 2, 3]) >= self.threshold
        self.tp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_patch, y_pred_patch), tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_patch, tf.logical_not(y_pred_patch)), tf.float32)))

    def result(self):
        """Return recall over all labeled patches seen by this metric instance."""
        return self.tp / tf.maximum(1.0, self.tp + self.fn)

    def reset_state(self):
        """Clear accumulated patch counts between epochs."""
        self.tp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        """Store the detection threshold in saved Keras metric config."""
        cfg = super().get_config()
        cfg.update({"threshold": self.threshold})
        return cfg


def make_loss_and_metrics(cfg):
    """
    Construct recall-friendly segmentation loss and metrics for training.
    The returned metric list includes pixel-level and patch-level thresholds.
    """
    lcfg = _loss_cfg(cfg)
    name = str(lcfg.get("name", "binary_crossentropy")).lower()
    alpha = float(lcfg.get("tversky_alpha", 0.3))
    beta = float(lcfg.get("tversky_beta", 0.7))
    pos_weight = float(lcfg.get("positive_weight", lcfg.get("pos_weight", 1.0)))
    bce_weight = float(lcfg.get("bce_weight", 0.5))
    tversky_weight = float(lcfg.get("tversky_weight", 0.5))

    bce = keras.losses.BinaryCrossentropy(from_logits=False)
    if name in ("binary_crossentropy", "bce"):
        loss = bce
    elif name == "weighted_bce":
        loss = WeightedBCELoss(positive_weight=pos_weight)
    elif name == "tversky":
        loss = TverskyLoss(alpha=alpha, beta=beta)
    elif name in ("bce_tversky", "weighted_bce_tversky"):
        loss = BCETverskyLoss(
            alpha=alpha,
            beta=beta,
            bce_weight=bce_weight,
            tversky_weight=tversky_weight,
            positive_weight=pos_weight,
            weighted_bce=name.startswith("weighted"),
            name=name,
        )
    else:
        raise ValueError(f"Unknown loss.name: {name}")

    thresholds = cfg.get("metrics", {}).get("thresholds", [0.05, 0.075, 0.1])

    metrics = [
        keras.metrics.AUC(curve="PR", name="pr_auc"),
    ]
    for threshold in thresholds:
        t = float(threshold)
        tag = str(t).replace(".", "p")
        metrics.extend(
            [
                keras.metrics.Precision(thresholds=t, name=f"pixel_precision_{tag}"),
                keras.metrics.Recall(thresholds=t, name=f"pixel_recall_{tag}"),
                PatchPrecision(threshold=t, name=f"patch_precision_{tag}"),
                PatchRecall(threshold=t, name=f"patch_recall_{tag}"),
            ]
        )

    return loss, metrics
