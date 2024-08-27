import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
import matplotlib.pyplot as plt

explore_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True
)
# 取第0张图片
y = explore_data[0]

x = explore_data[0][0]

xx = explore_data.data[0]
plt.imshow(x, cmap='binary')
plt.show()
tmp = 0

transform_funcs = Compose([
    ToTensor(),
    Normalize((0.1307, ), (0.3081, ))  # 标准化，手写数字数据集的通用参数
])

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform_funcs
)
test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform_funcs
)

print(train_data.data.shape)
print(test_data.data.shape)



