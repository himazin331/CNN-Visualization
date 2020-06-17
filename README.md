# CNN-Visualization
Filter and Feature map Visualization

詳細は[こちら](https://qiita.com/hima_zin331/items/05c4a6a04e2f42300371)
*Not supported except in Japanese Language

___

## feature_visual.py

**How to use**

```
    # レイヤー情報を取得
    feature_vi = []
    feature_vi.append(model_vi.get_layer('input_1'))
    feature_vi.append(model_vi.get_layer('conv2d'))
    feature_vi.append(model_vi.get_layer('max_pooling2d'))
    feature_vi.append(model_vi.get_layer('conv2d_1'))
    feature_vi.append(model_vi.get_layer('max_pooling2d_1'))
```

Specify the name of the feature map to be visualized in `model_vi.get_layer()`.
As appropriate, increase or decrease the number of `model_vi.get_layer()` by the number of feature maps you want to visualize.
You can find out the name of a layer by using model.summary().

## filter_visual.py

**How to use**

```
    # 可視化対象レイヤー
    vi_layer.append(model.get_layer('conv2d'))
    vi_layer.append(model.get_layer('conv2d_1'))
    vi_layer.append(model.get_layer('conv2d_2'))
```

Specify the name of the filter to be visualized in `model.get_layer()`.
As appropriate, increase or decrease the number of `model.get_layer()` by the number of filter you want to visualize.
