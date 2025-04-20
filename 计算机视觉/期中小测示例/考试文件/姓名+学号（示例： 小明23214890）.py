########################
#                      #
#       test1          #
#                      #
########################
import torch

# 定义输入向量 a 和 b（这里用随机值初始化，实际使用时替换为具体值）
a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], requires_grad=True)
b = torch.tensor([0.01] * 10, requires_grad=True)

#在这里实现
pass

# 手动验证梯度公式
# 1. 对 a 的梯度 = -softmax(a)
softmax_a = torch.exp(a) / sum_exp_a
grad_a_manual = -softmax_a

# 2. 对 b 的梯度 = 1/sum_b - 1
grad_b_manual = torch.ones_like(b) * (1 / sum_b - 1)

# 打印结果
print("自动微分计算的梯度:")
print("Gradient wrt a:\n", grad_a)
print("Gradient wrt b:\n", grad_b)

print("\n手动公式验证的梯度:")
print("Gradient wrt a (manual):\n", grad_a_manual)
print("Gradient wrt b (manual):\n", grad_b_manual)

# 检查一致性
assert torch.allclose(grad_a, grad_a_manual, atol=1e-4), "a的梯度验证失败!"
assert torch.allclose(grad_b, grad_b_manual, atol=1e-4), "b的梯度验证失败!"
print("\n梯度验证通过!")



########################
#                      #
#       test2          #
#                      #
########################
import cv2
import numpy as np
import matplotlib.pyplot as plt
#锐化卷积核
kernel1 = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

#边缘检测卷积核（垂直边缘）
kernel2 = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

#高斯模糊卷积核
kernel3 = np.array([
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
], dtype=np.float32)
def Filter(kernel,img,row,col):
    result=0
    # 卷积核的边⻓
    dim=np.array(kernel).shape[0]
    for i in range(dim):
        for c in range(dim):
            result=result+kernel [i][c]*img [i+row][col+c]
    return result
def Conv(kernel,img,res_row,res_col):
    Result = np.zeros((res_row,res_col))
    # 通过调⽤前⾯的Filter函数来对特征图每⼀个位置进⾏卷积操作
    for i in range(res_row):
        for c in range(res_col):
            Result [i][c]=Filter(kernel,img,i,c)
    return Result

# 读取并处理图像
img = cv2.imread('fig1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
n=________________________  # 补全1
width, height=________________________# 补全2
img_resized = cv2.resize(img,（width, height）, interpolation=cv2.INTER_LINEAR)
img_pad = np.pad(img_resized,n, mode='constant')

# 创建2x2的子图布局
plt.figure(figsize=(10, 10))

# 显示原始图像
plt.subplot(2, 2, 1)
plt.imshow(img_pad, cmap='gray')
plt.title('Original Image')

# 计算并显示锐化结果
featuremap_pad1 = Conv(kernel1, img_pad, 50, 50)
plt.subplot(2, 2, 2)
plt.imshow(featuremap_pad1, cmap='gray')
plt.title('Sharpened')

# 计算并显示边缘检测结果
featuremap_pad2 = Conv(kernel2, img_pad, 50, 50)
plt.subplot(2, 2, 3)
plt.imshow(featuremap_pad2, cmap='gray')
plt.title('Edge Detection')

# 计算并显示高斯模糊结果
featuremap_pad3 = Conv(kernel3, img_pad, 50, 50)
plt.subplot(2, 2, 4)
plt.imshow(featuremap_pad3, cmap='gray')
plt.title('Gaussian Blur')

# 调整子图间距
plt.tight_layout()

# 显示所有图像
plt.show()

########################
#                      #
#       test3          #
#                      #
########################
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def softmax(x):
    """计算输入x的softmax函数"""
    x_stable = x - np.max(x, axis=1, keepdims=True)  #
    softmax_prob = ________________________  # 补全1: softmax值
    return softmax_prob


def train_softmax(X_train, y_train, num_classes, learning_rate=0.01, epochs=100):
    """
    训练Softmax回归模型

    参数:
    X_train - 训练数据 (N, D)=(1437 ,64)
    y_train - 训练标签 (N,)
    num_classes - 类别数量
    learning_rate - 学习率
    epochs - 训练轮数

    返回:
    W - 训练好的权重矩阵 (D, C)=(64, 10)
    """
    N, D = X_train.shape  # (1437 ,64)
    C = num_classes  # 10

    # 初始化权重矩阵
    W = np.random.randn(D, C) * 0.01

    # 将标签转换为one-hot编码
    y_one_hot = np.zeros((N, C))
    y_one_hot[np.arange(N), y_train] = 1

    for epoch in range(epochs):
        # 前向传播
        scores = ________________________  # 补全2: 计算得分
        # 提示: 得分是输入数据X_train与权重矩阵W的线性组合
        # 使用矩阵乘法运算符@或np.dot()
        # scores.shape=(1437, 10)
        probs = softmax(scores)
        # 计算损失 (交叉熵损失)
        loss = ________________________  # 补全3: 计算损失

        # 反向传播，计算梯度
        grad = _
        X_train.T @ (probs - y_one_hot) / N  # 补全4: 计算梯度

        # 更新权重
        W = ________________________  # 补全5: 更新权重

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return W


# 加载数据
digits = load_digits()
X = digits.data
y = digits.target

# 数据预处理
X = X / 16.0  # 归一化到0-1范围

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
W = train_softmax(X_train, y_train, num_classes=10, learning_rate=0.2, epochs=200)

# 测试模型
scores_test = X_test @ W
predicted_classes = np.argmax(scores_test, axis=1)
accuracy = np.mean(predicted_classes == y_test)
print(f"Test Accuracy: {accuracy:.4f}")

########################
#                      #
#       test4          #
#                      #
########################
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def softmax(x):
    """计算输入x的softmax函数"""
    x_stable = x - np.max(x, axis=1, keepdims=True)  # 数值稳定性处理
    exp_x = np.exp(x_stable)
    softmax_prob = ________________________  # 补全1: softmax值
    return softmax_prob


def train_softmax_Mini_batch_sgd(X_train, y_train, num_classes, learning_rate=0.01, epochs=100):
    """
    训练Softmax回归模型（使用SGD优化）

    参数:
    X_train - 训练数据 (N, D)=(1437 ,64)
    y_train - 训练标签 (N,)
    num_classes - 类别数量
    learning_rate - 学习率
    epochs - 训练轮数

    返回:
    W - 训练好的权重矩阵 (D, C)=(64, 10)
    """
    N, D = X_train.shape  # (1437 ,64)
    C = num_classes  # 10

    # 初始化权重矩阵
    W = np.random.randn(D, C) * 0.01

    # 将标签转换为one-hot编码
    y_one_hot = np.zeros((N, C))
    y_one_hot[np.arange(N), y_train] = 1

    for epoch in range(epochs):
        # 随机打乱数据
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        y_one_hot_shuffled = y_one_hot[indices]

        # 小批量梯度下降（Mini-batch SGD）
        batch_size = 32
        for i in range(0, N, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_one_hot[i:i + batch_size]

            # 前向传播
            scores = ________________________  # 补全2: 计算得分
            probs = softmax(scores)

            # 计算损失（仅用于监控）
            loss = ________________________  # 补全3: 计算单个样本的损失

            # 反向传播，计算梯度
            grad = ________________________  # 补全4: 计算梯度

            # 更新权重
            W = ________________________  # 补全5: 更新权重

        # 每10轮打印一次整个训练集的损失
        if epoch % 10 == 0:
            scores_full = X_train @ W
            probs_full = softmax(scores_full)
            full_loss = -np.mean(np.sum(y_one_hot * np.log(probs_full + 1e-8), axis=1))
            print(f"Epoch {epoch}, Loss: {full_loss:.4f}")

    return W


# 加载数据
digits = load_digits()
X = digits.data
y = digits.target

# 数据预处理
X = X / 16.0  # 归一化到0-1范围

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型（使用Mini-batch SGD）
W = train_softmax_Mini_batch_sgd(X_train, y_train, num_classes=10, learning_rate=0.2, epochs=200)

# 测试模型
scores_test = X_test @ W
predicted_classes = np.argmax(scores_test, axis=1)
accuracy = np.mean(predicted_classes == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
