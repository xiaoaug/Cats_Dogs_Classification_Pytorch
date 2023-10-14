识别猫狗图片，使用 pytorch 实现。

训练 20 epoch，准确率为 90.88%，且继续训练准确率仍旧可以提高。

数据集下载链接在 train_set 和 test_set 文件夹内的 readme.txt 中。

<img src="https://github.com/xiaoaug/Cats_Dogs_Classification_Pytorch/assets/39291338/f4494016-e7e7-44b4-aa89-1349fa40dac2" width="250">

本项目测试环境为 Ubuntu20.04，python 版本为 3.10.13。

# 如何安装？
```
git clone https://github.com/xiaoaug/Cats_Dogs_Classification_Pytorch.git  # 下载
cd Cats_Dogs_Classification_Pytorch
pip install -r requirements.txt  # 安装
```

# 如何训练？

1. 运行程序前，需将数据集下载下来并按照要求放置于 train_set 和 test_set 文件夹下。
2. 根据自己的需要修改 train.py 文件中第 11~23 行的参数（默认也可以）。
3. 运行 train.py 即可：`python train.py`。

> 该项目每轮训练中，只要训练准确率比之前高，就会生成 pth 文件。若该轮训练的准确率比以往训练的准确率低，则不生成 pth 文件。

# 如何预测？

1. 根据自己的需要修改 predict.py 文件中第 11~17 行的参数（默认也可以）。
2. 运行 predict.py 即可：`python predict.py`。

