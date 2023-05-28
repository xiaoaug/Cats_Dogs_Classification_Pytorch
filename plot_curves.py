from matplotlib import pyplot as plt


def plot_curves(results: dict[str, list]) -> None:
    """
    绘制模型的训练结果曲线
    :param results: 训练结果
    :return: None
    """
    print('----> Drawing Curve')

    train_loss = results['train_loss']
    test_loss = results['test_loss']
    train_accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    epochs = range(1, len(results['train_loss'])+1)  # 有多少个损失数据就有多少个 epoch

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train_Loss')
    plt.plot(epochs, test_loss, label='Test_Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Train_Accuracy')
    plt.plot(epochs, test_accuracy, label='Test_Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()
    print('----> Done')


if __name__ == '__main__':
    test = {'train_loss': [1.0, 2.0, 3.0], 'test_loss': [1.1, 2.1, 3.1],
            'train_acc': [2.0, 3.0, 4.0], 'test_acc': [2.1, 3.1, 4.1]}
    plot_curves(test)
