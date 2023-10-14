import os
import torch
from torch import nn
from tqdm import tqdm

import cnn_model
import dataset
import plot_curves

# setting
DATA_DIR = os.getcwd()  # 程序文件路径
TRAIN_DIR = DATA_DIR + "/train_set"  # 训练集路径
TEST_DIR = DATA_DIR + "/test_set"  # 测试集路径
BATCH_SIZE = 32  # 一次训练的样本量
NUM_WORKERS = 2  # 有多少子进程将用于数据的加载，若使用 Windows 建议设置为 0
NUM_EPOCHS = 2  # 训练轮数
LEARNING_RATE = 1e-3  # 训练学习率
CONTINUE_TRAIN = False  # 是否使用以前的 pth 文件继续训练？
PTH_FILE = f"{DATA_DIR}/checkpoint_90.8831%.pth"  # 调用之前生成好的 pth 文件并继续训练
PTH_SAVE_DIR = DATA_DIR  # pth 文件保存路径
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_ACC = 0.0  # 最佳正确率

print("----> Creating Model")
my_model = cnn_model.CNN().to(DEVICE)
loss_fn = nn.CrossEntropyLoss().to(DEVICE)  # 损失函数
optim = torch.optim.Adam(params=my_model.parameters(), lr=LEARNING_RATE)  # 优化器
print("----> Done")

if CONTINUE_TRAIN:
    print("----> Loading Checkpoint")
    my_model.load_state_dict(torch.load(PTH_FILE))
    print("----> Done")


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    curr_epoch: int,
) -> tuple[float, float]:
    """
    单轮训练
    :param model: 模型
    :param dataloader: 训练数据集
    :param loss_function: 损失函数
    :param optimizer: 优化器
    :param curr_epoch: 当前训练的轮数
    :return: [平均损失, 平均准确度]
    """
    model.train()
    train_loss, train_acc = 0.0, 0.0

    t = tqdm(enumerate(dataloader, start=1), total=len(dataloader))  # 进度条

    for batch_idx, (image, label) in t:
        image, label = image.to(DEVICE), label.to(DEVICE)  # 将数据发送到目标设备
        pred_label = model(image)  # 前向传递
        loss = loss_function(pred_label, label)  # 计算损失
        train_loss += loss.item()  # 累计损失
        optimizer.zero_grad()  # 优化器清零
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新

        # 计算准确度
        pred_class = torch.argmax(
            torch.softmax(pred_label, dim=1), dim=1
        )  # softmax 处理完后取最大索引，即为类别
        train_acc += (pred_class == label).sum().item() / len(pred_label)

        t.set_description(f"[Train Epoch = {curr_epoch}/{NUM_EPOCHS}]")
        t.set_postfix(
            Train_Loss=f"{train_loss/batch_idx:.4f}",
            Train_Acc=f"{train_acc/batch_idx:.4f}",
        )

    # 计算平均损失和准确度
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    curr_epoch: int,
) -> tuple[float, float]:
    """
    单轮验证测试
    :param model: 模型
    :param dataloader: 测试数据集
    :param loss_function: 损失函数
    :param curr_epoch: 当前训练的轮数
    :return: [平均损失, 平均准确度]
    """
    model.eval()
    test_loss, test_acc = 0.0, 0.0

    t = tqdm(enumerate(dataloader, start=1), total=len(dataloader))  # 进度条

    with torch.inference_mode():
        for batch_idx, (image, label) in t:
            image, label = image.to(DEVICE), label.to(DEVICE)  # 将数据发送到目标设备
            pred_label = model(image)  # 前向传播
            loss = loss_function(pred_label, label)  # 计算损失
            test_loss += loss.item()  # 累计损失

            # 计算准确度
            pred_class = pred_label.argmax(dim=1)
            test_acc += (pred_class == label).sum().item() / len(pred_class)

            t.set_description(f"[Test  Epoch = {curr_epoch}/{NUM_EPOCHS}]")
            t.set_postfix(
                Test_Loss=f"{test_loss / batch_idx:.4f}",
                Test_Acc=f"{test_acc / batch_idx:.4f}",
            )

    # 计算平均损失和准确度
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module = nn.CrossEntropyLoss(),
) -> dict[str, list]:
    """
    训练
    :param model: 模型
    :param train_dataloader: 训练数据集
    :param test_dataloader: 测试数据集
    :param optimizer: 优化器
    :param loss_function: 损失函数
    :return: dict{ 训练平均损失, 训练准确度, 测试平均损失, 测试准确度 }
    """
    global BEST_ACC
    # 创建空结果字典，用于后期 plt 绘图
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    print("----> Start Training")

    # 循环执行训练和测试
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            curr_epoch=epoch,
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_function=loss_function,
            curr_epoch=epoch,
        )

        # 生成 pth 文件
        if test_acc > BEST_ACC:
            BEST_ACC = test_acc
            torch.save(
                model.state_dict(),
                f"{PTH_SAVE_DIR}/checkpoint_{test_acc*100:.4f}%.pth",
            )

        # 更新结果字典
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    print("----> Done")
    return results


if __name__ == "__main__":
    model_results = train(
        model=my_model,
        train_dataloader=dataset.get_train_data_loader(),
        test_dataloader=dataset.get_test_data_loader(),
        optimizer=optim,
        loss_function=loss_fn,
    )
    plot_curves.plot_curves(model_results)
