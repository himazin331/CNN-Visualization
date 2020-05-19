import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TFメッセージ非表示

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# 特徴マップ可視化
def feature_vi(model, input_shape, test_img):
        
    # モデル再構築
    x = tf.keras.Input(shape=input_shape)
    model_vi = tf.keras.Model(inputs=x, outputs=model.call(x))
     
    # ネットワーク構成出力
    model_vi.summary()
    print("")
        
    # レイヤー情報を取得
    feature_vi = []
    feature_vi.append(model_vi.get_layer('input_1'))
    feature_vi.append(model_vi.get_layer('conv2d'))
    feature_vi.append(model_vi.get_layer('max_pooling2d'))
    feature_vi.append(model_vi.get_layer('conv2d_1'))
    feature_vi.append(model_vi.get_layer('max_pooling2d_1'))

    # データランダム抽出
    idx = int(np.random.randint(0, len(test_img), 1))
    img = test_img[idx]
    img = img[None, :, :, :]

    for i in range(len(feature_vi)-1):
            
        # 特徴マップ取得
        feature_model = tf.keras.Model(inputs=feature_vi[0].input, outputs=feature_vi[i+1].output)
        feature_map = feature_model.predict(img)
        feature_map = feature_map[0]
        feature = feature_map.shape[2]
            
        # ウィンドウ名定義
        fig = plt.gcf()
        fig.canvas.set_window_title(feature_vi[i+1].name + " feature-map visualization")
            
        # 出力
        for j in range(feature):
            plt.subplots_adjust(wspace=0.4, hspace=0.8)
            plt.subplot(feature/6 + 1, 6, j+1)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f'filter {j}')
            plt.imshow(feature_map[:,:,j])
        plt.show()