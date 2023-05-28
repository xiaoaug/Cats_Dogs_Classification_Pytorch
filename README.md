识别猫狗图片，使用 pytorch 实现。

训练 20 epoch，准确率为 90.88%，且继续训练准确率仍旧可以提高。

数据集下载链接在 train_set 和 test_set 文件夹内的 readme.txt 中。

<img src="https://github.com/xiaoaug/Cats_Dogs_Classification_Pytorch/assets/39291338/f4494016-e7e7-44b4-aa89-1349fa40dac2" width="250">

# 如何训练？

1. 运行程序前，需将 train_set 和 test_set 文件夹下的压缩包解压。
2. 在 setting.py 中根据你自己的情况修改参数。
3. 运行 train.py 即可。

# 如何预测？

1. 在 setting.py 中根据你自己的情况修改参数。
2. 在 predict.py 中将第 12 行的 `PRED_IMAGE_NAME` 进行修改，换成你需要预测的图片名称。
3. 运行 predict.py 即可。

