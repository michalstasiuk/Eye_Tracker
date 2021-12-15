from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.backend import random_normal_variable
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import MaxPooling1D, Input, concatenate, Conv2D , MaxPooling2D, BatchNormalization , DepthwiseConv2D, LeakyReLU
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def et_base():

    regularization = 1e-11

    model_in = Input(shape=(64,64,3))
    model_out = Conv2D(32, 3,padding='same', input_shape=(64,64,3),
                       kernel_initializer="he_normal", activity_regularizer=l1_l2(regularization,regularization))(model_in)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = MaxPooling2D()(model_out)
    model_out = Conv2D(32, 3 ,padding='same', kernel_initializer="he_normal",activity_regularizer=l1_l2(regularization,regularization))(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = BatchNormalization()(model_out)
    model_out = DepthwiseConv2D(7, padding='same', dilation_rate=1)(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = Dropout(0.1)(model_out)

    model_out = Conv2D(64, 3 ,padding='same', kernel_initializer="he_normal",activity_regularizer=l1_l2(regularization,regularization))(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = Conv2D(64, 3 ,padding='same', kernel_initializer="he_normal",activity_regularizer=l1_l2(regularization,regularization))(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = BatchNormalization()(model_out)
    model_out = DepthwiseConv2D(5, padding='same')(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = Dropout(0.1)(model_out)

    model_out = Conv2D(64, 3 ,padding='same', kernel_initializer="he_normal",activity_regularizer=l1_l2(regularization,regularization))(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = BatchNormalization()(model_out)
    model_out = DepthwiseConv2D(5, padding='same')(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = Dropout(0.1)(model_out)

    model_out = Conv2D(64, 3 ,padding='same', kernel_initializer="he_normal",activity_regularizer=l1_l2(regularization,regularization))(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = BatchNormalization()(model_out)
    model_out = DepthwiseConv2D(5, padding='same')(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = Dropout(0.1)(model_out)

    model_out = Conv2D(128, 3,padding='same', kernel_initializer="he_normal",activity_regularizer=l1_l2(regularization,regularization))(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = MaxPooling2D()(model_out)
    model_out = BatchNormalization()(model_out)
    model_out = DepthwiseConv2D(5, padding='same')(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = Dropout(0.1)(model_out)

    model_out = Conv2D(128, 3 ,padding='same', kernel_initializer="he_normal",activity_regularizer=l1_l2(regularization,regularization))(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = BatchNormalization()(model_out)
    model_out = DepthwiseConv2D(5, padding='same')(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = Dropout(0.1)(model_out)

    model_out = Conv2D(256, 3, strides=(2,2) ,padding='same', kernel_initializer="he_normal",activity_regularizer=l1_l2(regularization,regularization))(model_out)
    model_out = MaxPooling2D()(model_out)
    model_out = BatchNormalization()(model_out)
    model_out = DepthwiseConv2D(5, padding='same')(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = Dropout(0.1)(model_out)

    model_out = Conv2D(256, 3 ,padding='same', kernel_initializer="he_normal",activity_regularizer=l1_l2(regularization,regularization))(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = BatchNormalization()(model_out)
    model_out = DepthwiseConv2D(5, padding='same')(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = Dropout(0.1)(model_out)

    model_out = Flatten()(model_out)

    return model_in, model_out

def et_high_level():
    model_in = Input(shape=(6))
    model_out = Dense(32, activation="relu")(model_in)
    model_out = Dense(32, activation="relu")(model_out)
    model_out = Dense(16, activation="relu")(model_out)

    return model_in, model_out

def et_face_landmarks():
    model_in = Input(shape=(456,3))
    model_out = Conv1D(32, 5)(model_in)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = MaxPooling1D(2)(model_out)
    model_out = Conv1D(64, 5)(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = MaxPooling1D(2)(model_out)
    model_out = Conv1D(128, 5)(model_out)
    model_out = LeakyReLU(alpha=0.01)(model_out)
    model_out = MaxPooling1D(2)(model_out)
    model_out = Flatten()(model_out)
    model_out = Dense(512, activation="relu")(model_out)
    model_out = Dense(512, activation="relu")(model_out)
    model_out = Dense(16, activation="relu")(model_out)

    return model_in, model_out

def eye_tracker():

    model_eye_in, model_eye_out = et_base()
    model_nose_in, model_nose_out = et_base()
    model_high_lvl_in, model_high_lvl_out = et_high_level()

    low_level = concatenate([model_eye_out, model_nose_out])
    low_level = Dense(512)(low_level)
    low_level = LeakyReLU(alpha=0.01)(low_level)
    low_level = Dropout(0.1)(low_level)
    low_level = Dense(512)(low_level)
    low_level = LeakyReLU(alpha=0.01)(low_level)
    low_level = Dropout(0.1)(low_level)
    low_level = Dense(32)(low_level)
    low_level = LeakyReLU(alpha=0.01)(low_level)
    low_level = Dropout(0.1)(low_level)

    concatenated = concatenate([low_level, model_high_lvl_out])
    concatenated = Dense(1)(concatenated)

    merged_model = Model([model_eye_in, model_nose_in, model_high_lvl_in], concatenated)
    
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    merged_model.compile(optimizer=optimizer, loss='mean_absolute_error')

    merged_model.summary()

    tf.keras.utils.plot_model(merged_model)

    return merged_model

def eye_tracker_simple():
    model = Sequential()
    model.add(Conv2D(32, 3, strides=(2,2),padding='same', input_shape=(64,64,3)))
    model.add(Conv2D(32, 3 ,padding='same'))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(7, padding='same', dilation_rate=1))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, 3 ,padding='same'))
    model.add(Conv2D(64, 3 ,padding='same'))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(5, padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, 3 ,padding='same'))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(5, padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, 3 ,padding='same'))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(5, padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, 3, strides=(2,2) ,padding='same'))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(5, padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, 3 ,padding='same'))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(5, padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(256, 3, strides=(2,2) ,padding='same'))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(5, padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(256, 3 ,padding='same'))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(5, padding='same'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(512))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    model.summary()

    return model
