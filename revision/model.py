"""Revised LSTM-VAE classifier (wrapper around qubit_ablation_study.LSTM_VAE_Model).

The layer stack is IDENTICAL to the original ``LSTM_VAE_Model.build`` (LSTM(u, seq) ->
LayerNorm -> MHA(heads, key_dim=u/heads) + residual -> LayerNorm -> LSTM(u/2) -> LayerNorm ->
Dense z_mean / Dense z_log_var -> z -> Dense(64, relu) -> Dropout -> Dense(32, relu) ->
Dense(C, softmax)).  Two fixes (coordinator addendum):

  1. The original "VAE" had no KL / reconstruction term (loss = CE only).  We add a KL
     regulariser via ``add_loss``:  gamma * mean_batch(-0.5 * sum_latent(1 + lv - mu^2 - exp(lv))),
     gamma default 1e-3.  No reconstruction decoder; CE stays the main loss.
  2. The original Lambda sampling layer sampled at inference as well (non-deterministic
     predictions).  ``Reparameterize`` samples only when training=True and returns z_mean
     otherwise.

``compile_with_sgd`` is inherited unchanged: SGD(lr, momentum=0.9, nesterov=True),
categorical cross-entropy, accuracy.
"""
from __future__ import annotations

from typing import Tuple

import common  # noqa: F401  (adds repo root to sys.path)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, LayerNormalization,
                                     MultiHeadAttention)
from tensorflow.keras.models import Model

from qubit_ablation_study import LSTM_VAE_Model  # original (unmodified) class

DEFAULT_MODEL_CFG = dict(lstm_units=256, attention_heads=2, dropout_rate=0.2, latent_dim=64)


class Reparameterize(keras.layers.Layer):
    """z = mu + exp(0.5 lv) * eps when training, z = mu at inference; adds gamma * KL."""

    def __init__(self, kl_weight: float = 1e-3, **kw):
        super().__init__(**kw)
        self.kl_weight = float(kl_weight)

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs
        if self.kl_weight > 0:
            kl = -0.5 * tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            self.add_loss(self.kl_weight * tf.reduce_mean(kl))
        if training:
            eps = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * eps
        return z_mean

    def get_config(self):
        return {**super().get_config(), "kl_weight": self.kl_weight}


class RevisedLSTMVAE(LSTM_VAE_Model):
    def __init__(self, input_shape: Tuple[int, int], num_classes: int, lstm_units: int = 256,
                 attention_heads: int = 2, dropout_rate: float = 0.2, latent_dim: int = 64,
                 kl_weight: float = 1e-3):
        super().__init__(input_shape=input_shape, num_classes=num_classes, lstm_units=lstm_units,
                         attention_heads=attention_heads, dropout_rate=dropout_rate, latent_dim=latent_dim)
        self.kl_weight = kl_weight

    def build(self) -> Model:  # identical stack except the sampling layer
        inputs = Input(shape=self.input_shape)
        x = LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate)(inputs)
        x = LayerNormalization()(x)
        attention_output = MultiHeadAttention(num_heads=self.attention_heads,
                                              key_dim=self.lstm_units // self.attention_heads)(x, x)
        x = x + attention_output
        x = LayerNormalization()(x)
        x = LSTM(self.lstm_units // 2, return_sequences=False, dropout=self.dropout_rate)(x)
        x = LayerNormalization()(x)
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        z = Reparameterize(kl_weight=self.kl_weight, name="z")([z_mean, z_log_var])
        x = Dense(64, activation="relu")(z)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(self.num_classes, activation="softmax")(x)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.encoder = Model(inputs=inputs, outputs=[z_mean, z_log_var, z])
        return self.model


def build_model(input_shape: Tuple[int, int], num_classes: int = 2, learning_rate: float = 1e-3,
                kl_weight: float = 1e-3, **cfg) -> RevisedLSTMVAE:
    c = {**DEFAULT_MODEL_CFG, **cfg}
    m = RevisedLSTMVAE(input_shape, num_classes, kl_weight=kl_weight, **c)
    m.build()
    m.compile_with_sgd(learning_rate=learning_rate)
    return m


def count_params(model: Model) -> dict:
    import numpy as np

    tr = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    nt = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    return {"trainable": tr, "non_trainable": nt, "total": tr + nt}


def optimizer_summary(model: Model) -> dict:
    opt = model.optimizer
    cfg = opt.get_config()
    return {k: cfg.get(k) for k in ("name", "learning_rate", "momentum", "nesterov", "weight_decay",
                                    "clipnorm", "clipvalue", "global_clipnorm")}
