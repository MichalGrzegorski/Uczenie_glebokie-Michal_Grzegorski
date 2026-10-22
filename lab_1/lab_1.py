# LAB01

import math
import os
import time

import keras
import matplotlib as plt
import numpy as np
import tensorflow as tf
import torch

# ZADANIE 1.1
print("\n#####   ZADANIE 1.1   #####\n")

rng = np.random.default_rng(0)
batch_np = rng.normal(
    loc=[0.5, 0.2, -0.3], scale=[2.0, 0.5, 1.0], size=(16, 32, 32, 3)
).astype(np.float32)

# TensorFlow (NHWC)
batch_tf = tf.constant(batch_np)
mean_tf = tf.reduce_mean(batch_tf, axis=[0, 1, 2], keepdims=True)
std_tf = tf.math.reduce_std(batch_tf, axis=[0, 1, 2], keepdims=True)
z_tf = (batch_tf - mean_tf) / std_tf

# PyTorch (NCHW)
batch_pt = torch.from_numpy(batch_np).permute(0, 3, 1, 2)
mean_pt = batch_pt.mean(dim=(0, 2, 3), keepdim=True)
std_pt = batch_pt.std(dim=(0, 2, 3), correction=0, keepdim=True)
z_pt = (batch_pt - mean_pt) / std_pt


# Komórka sprawdzająca
if z_tf is None or z_pt is None:
    print("Uzupełnij z_tf i z_pt.")
else:
    print(
        "TF  średnie per kanał:",
        np.round(tf.reduce_mean(z_tf, axis=[0, 1, 2]).numpy(), 4),
    )
    print(
        "TF  std per kanał    :",
        np.round(tf.math.reduce_std(z_tf, axis=[0, 1, 2]).numpy(), 4),
    )
    print("PT  średnie per kanał:", np.round(z_pt.mean(dim=(0, 2, 3)).numpy(), 4))
    z_pt_nhwc = z_pt.permute(0, 2, 3, 1).numpy()  # powrót do NHWC
    print("Zgodność TF vs PyTorch:", np.allclose(z_tf.numpy(), z_pt_nhwc, atol=1e-5))


# ZADANIE 1.2
print("\n#####   ZADANIE 1.1   #####\n")
