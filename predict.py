import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

import cnn_model
import setting

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['cats', 'dogs']   # 预测猫和狗
PRED_IMAGE_NAME = 'dog.1.jpg'    # 用于预测的图像文件名
PRED_IMAGE = f"{setting.PRED_DIR}/{PRED_IMAGE_NAME}"  # pred_set 文件夹内的图片，用于预测图像

print('----> Creating Model')
my_model = cnn_model.CNN().to(DEVICE)
print('----> Done')
print("----> Loading Checkpoint")
my_model.load_state_dict(torch.load(setting.PTH_FILE, map_location=DEVICE))
print("----> Done")


def get_image():
    """
    读取用于预测的图片，并进行相应的处理
    :return: 处理完成的图片
    """
    print("----> Loading Image")
    img = torchvision.io.read_image(PRED_IMAGE).type(torch.float32)  # 读取图片
    print("----> Image Shape: ", img.shape)
    img = img / 255.0  # 将图像像素值除以 255 使其数据在 [0, 1] 之间
    print("----> Image Resize To: ", setting.IMAGE_SIZE)
    resize = transforms.Resize(size=setting.IMAGE_SIZE, antialias=True)  # 调整图片尺寸
    img = resize(img)
    print("----> Image Shape: ", img.shape)
    print("----> Done")

    return img


def predict() -> None:
    """
    预测
    :return: None
    """
    my_model.eval()
    with torch.inference_mode():
        image = get_image()  # 获取图片
        print("----> Start Predicting")
        pred_label = my_model(image.unsqueeze(dim=0).to(DEVICE))
    # print("----> Predict Label: ", pred_label)

    pred_label_softmax = torch.softmax(pred_label, dim=1)     # 转换成概率
    pred_label_idx = torch.argmax(pred_label_softmax, dim=1)  # 获取概率最大的下标
    pred_label_class = CLASS_NAMES[pred_label_idx.cpu()]  # 找出是狗还是猫
    print(f"----> Predict: === {pred_label_class} ===")
    print("----> Done")

    # 打印图像
    print("----> Drawing Image")
    plt.imshow(image.permute(1, 2, 0))  # 需要把图像从 CHW 转成 HWC
    plt.title(f"Predict: {pred_label_class}")
    plt.axis(False)
    plt.show()
    print("----> Done")


if __name__ == '__main__':
    predict()
