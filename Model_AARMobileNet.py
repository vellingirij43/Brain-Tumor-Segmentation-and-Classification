import numpy as np
import tensorflow as tf
import cv2 as cv
from keras import layers, models
from Evaluation import evaluation


# Define Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def _init_(self, **kwargs):
        super(AttentionLayer, self)._init_(**kwargs)

    def build(self, input_shape):
        self.W_q = self.add_weight(name="W_q", shape=(input_shape[-1], input_shape[-1]), initializer="uniform",
                                   trainable=True)
        self.W_k = self.add_weight(name="W_k", shape=(input_shape[-1], input_shape[-1]), initializer="uniform",
                                   trainable=True)
        self.W_v = self.add_weight(name="W_v", shape=(input_shape[-1], input_shape[-1]), initializer="uniform",
                                   trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        q = tf.matmul(x, self.W_q)
        k = tf.matmul(x, self.W_k)
        v = tf.matmul(x, self.W_v)

        attn_scores = tf.matmul(q, k, transpose_b=True)
        attn_scores = tf.nn.softmax(attn_scores, axis=-1)
        output = tf.matmul(attn_scores, v)

        return output


# Create MobileNetV2 model with Attention Layer
def Model_AARMobileNet(train_data, train_target, test_data, test_target):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet')

    IMG_SIZE = [224, 224, 3]
    Feat = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Train_x = Feat.reshape(Feat.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat_1 = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Feat[i, :] = cv.resize(test_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Test_x = Feat.reshape(Feat.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    # base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential()
    model.add(base_model)
    # model.add(layers.GlobalAveragePooling2D())
    model.add(AttentionLayer())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(units=train_target.shape[1], activation='softmax'))
    model.summary()

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_x, train_target, steps_per_epoch=1, epochs=2, validation_data=(Test_x, test_target))
    pred = model.predict(Test_x)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred
