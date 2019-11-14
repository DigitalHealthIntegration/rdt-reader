# -*- coding: utf-8 -*-
"""Freeze red-blue line detection model to use for inference. Change modelToLoad variable and export_path as needed

Example:

        $ python freeze_line_detection.py

"""
import os
import numpy as np
import cv2
import keras.backend as K
import tensorflow as tf
import keras
import random
import keras.metrics
import train_blue_red


modelToLoad = './weights-latest_model_YCrCb_test.hdf5'
export_path = './tensorflow-yolov3/models/Flu_audere_line/2'


if __name__ == "__main__":
    keras.metrics.recall_m = train_blue_red.recall_m
    keras.metrics.precision_m = train_blue_red.precision_m
    keras.metrics.f1_m = train_blue_red.f1_m
    tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference
    model = tf.keras.models.load_model(modelToLoad,custom_objects={"f1_m":keras.metrics.f1_m,"precision_m":keras.metrics.precision_m,"recall_m":keras.metrics.recall_m})

    inputs = tf.saved_model.utils.build_tensor_info(model.inputs[0])


    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key

    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_image': model.inputs[0]},
            outputs={"predictions":model.outputs[0]})
