import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation,concatenate, Conv2DTranspose, BatchNormalization, Dropout
from keras.models import Model
from tensorflow.keras.utils import plot_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

class AttentionUNet:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def res_block(self, x, filters, size):
        conv = Conv2D(filters, size, activation='relu', padding='same')(x)
        conv = Conv2D(filters, size, activation='relu', padding='same')(conv)
        conv = Conv2D(filters, size, activation='relu', padding='same')(conv)
        skip = Conv2D(filters, (1, 1), padding='same')(x)
        out = tf.keras.layers.Add()([conv, skip])
        out = Activation('relu')(out)
        return out

    def attn(self, x, g, channels):
        theta_x = Conv2D(channels, (1, 1), strides=(2, 2), padding='same')(x)
        phi_g = Conv2D(channels, (1, 1), padding='same')(g)
        xg = tf.keras.layers.Add()([theta_x, phi_g])
        xg = Activation('relu')(xg)
        psi = Conv2D(1, (1, 1), padding='same')(xg)
        sig_xg = Activation('sigmoid')(psi)
        up = UpSampling2D((2, 2))(sig_xg)
        out = tf.keras.layers.multiply([up, x])
        return out

    def build_unet(self):
        inputs = Input(self.input_shape)

        d1 = self.res_block(inputs, 32, (3, 3))
        d2 = MaxPooling2D((2, 2))(d1)
        d2 = self.res_block(d2, 64, (3, 3))
        d3 = MaxPooling2D((2, 2))(d2)
        d3 = self.res_block(d3, 128, (3, 3))
        bn = MaxPooling2D((2, 2))(d3)

        bn = self.res_block(bn, 256, (3, 3))

        u3 = UpSampling2D((2, 2))(bn)
        skip = self.attn(d3, bn, 256)
        u3 = concatenate([u3, skip])
        u3 = self.res_block(u3, 128, (3, 3))

        u2 = UpSampling2D((2, 2))(u3)
        skip = self.attn(d2, u3, 128)
        u2 = concatenate([u2, skip])
        u2 = self.res_block(u2, 64, (3, 3))

        u1 = UpSampling2D((2, 2))(u2)
        skip = self.attn(d1, u2, 64)
        u1 = concatenate([u1, skip])
        u1 = self.res_block(u1, 64, (3, 3))

        outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(u1)

        model = Model(inputs, outputs)
        return model


input_shape = (128, 256, 3)  
attention_unet = AttentionUNet(input_shape)
model = attention_unet.build_unet()
model.summary()

