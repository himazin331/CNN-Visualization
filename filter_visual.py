import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TFメッセージ非表示


# フィルタ可視化
def filter_vi(model):
    vi_layer = []

    # 可視化対象レイヤー
    vi_layer.append(model.get_layer('conv2d'))
    vi_layer.append(model.get_layer('conv2d_1'))
    vi_layer.append(model.get_layer('conv2d_2'))

    for i in range(len(vi_layer)):
        # レイヤーのフィルタ取得
        target_layer = vi_layer[i].get_weights()[0]
        filter_num = target_layer.shape[3]

        # ウィンドウ名定義
        fig = plt.gcf()
        fig.canvas.set_window_title(vi_layer[i].name + " filter visualization")

        # 出力
        for j in range(filter_num):
            plt.subplots_adjust(wspace=0.4, hspace=0.8)
            plt.subplot(filter_num / 6 + 1, 6, j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f'filter {j}')
            plt.imshow(target_layer[:, :, 0, j], cmap="gray")
        plt.show()
