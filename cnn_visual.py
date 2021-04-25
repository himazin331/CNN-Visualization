import tensorflow as tf
import tensorflow.keras.layers as kl

import numpy as np

import feature_visual
import filter_visual

import argparse as arg
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TFメッセージ非表示


# CNN
class CNN(tf.keras.Model):
    def __init__(self, n_out, input_shape):
        super().__init__()

        self.conv1 = kl.Conv2D(16, 4, activation='relu', input_shape=input_shape)
        self.conv2 = kl.Conv2D(32, 4, activation='relu')
        self.conv3 = kl.Conv2D(64, 4, activation='relu')

        self.mp1 = kl.MaxPool2D((2, 2), padding='same')
        self.mp2 = kl.MaxPool2D((2, 2), padding='same')
        self.mp3 = kl.MaxPool2D((2, 2), padding='same')

        self.flt = kl.Flatten()

        self.link = kl.Dense(1024, activation='relu')
        self.link_class = kl.Dense(n_out, activation='softmax')

    def call(self, x):
        h1 = self.mp1(self.conv1(x))
        h2 = self.mp2(self.conv2(h1))
        h3 = self.mp3(self.conv3(h2))
        
        h4 = self.link(self.flt(h3))

        return self.link_class(h4)


# 学習
class trainer(object):
    def __init__(self, n_out, input_shape):
        self.model = CNN(n_out, input_shape)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=['accuracy'])

    def train(self, train_img, train_lab, batch_size, epochs, input_shape, test_img):
        # 学習
        self.model.fit(train_img, train_lab, batch_size=batch_size, epochs=epochs)

        print("___Training finished\n\n")

        # 特徴マップ可視化
        feature_visual.feature_vi(self.model, input_shape, train_img)
        # フィルタ可視化
        filter_visual.filter_vi(self.model)


def main():
    # コマンドラインオプション作成
    parser = arg.ArgumentParser(description='CNN Feature-map & Filter Visualization')
    parser.add_argument('--batch_size', '-b', type=int, default=256,
                        help='ミニバッチサイズの指定(デフォルト値=256)')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='学習回数の指定(デフォルト値=10)')
    args = parser.parse_args()

    # データセット取得、前処理
    (train_img, train_lab), (test_img, _) = tf.keras.datasets.mnist.load_data()
    train_img = tf.convert_to_tensor(train_img, np.float32)
    train_img /= 255
    train_img = train_img[:, :, :, np.newaxis]

    test_img = tf.convert_to_tensor(test_img, np.float32)
    test_img /= 255
    test_img = train_img[:, :, :, np.newaxis]

    # 学習開始
    print("___Start training...")

    input_shape = (28, 28, 1)

    Trainer = trainer(10, input_shape)
    Trainer.train(train_img, train_lab, batch_size=args.batch_size,
                    epochs=args.epoch, input_shape=input_shape, test_img=test_img)


if __name__ == '__main__':
    main()
