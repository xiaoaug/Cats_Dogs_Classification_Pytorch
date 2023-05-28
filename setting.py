import os

DATA_DIR = os.getcwd()                  # 程序文件路径
TRAIN_DIR = DATA_DIR + "/train_set"     # 训练集路径
TEST_DIR = DATA_DIR + "/test_set"       # 测试集路径
PRED_DIR = DATA_DIR + "/pred_set"       # 预测集路径

CONTINUE_TRAIN = False  # 是否使用以前的 pth 文件继续训练？

PTH_SAVE_DIR = DATA_DIR    # pth 文件保存路径
PTH_FILE = f'{DATA_DIR}/checkpoint_90.8831%.pth'  # 调用之前生成好的 pth 文件并继续训练

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

BATCH_SIZE = 32         # 一次训练的样本量
NUM_WORKERS = 2         # 有多少子进程将用于数据的加载，若使用 Windows 建议设置为 0
NUM_EPOCHS = 20         # 训练轮数
LEARNING_RATE = 1e-3    # 训练学习率


if __name__ == '__main__':
    print(CONTINUE_TRAIN)
