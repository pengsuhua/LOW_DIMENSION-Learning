import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .image_dataset import ImageDataset,ImageDataset_normal,ImageDataset_add_noise,ImageDataset_normal_add_nosie
from copy import deepcopy
from tqdm import tqdm


# plt.ion()

'''' fitter for training:clean !!'''
class Fitter:
    def __init__(self, **kwargs):
        self.args = kwargs
        self.plot_manager = kwargs.get('plot_manager', None)

        model = kwargs.get('model',None)
        # model = kwargs.get('model', 'mlp')
        if model == 'mlp':
            from fit.models.mlp import MLPModel
            self.model = MLPModel()
        elif model == 'resnet':
            # TODO
            assert False
            from fit.models.resnet import ResNet  # TODO
            self.model = ResNet()
        else:
            assert False

        # self.learning_rate = kwargs.get('learning_rate', 0.0001)
        # self.num_epochs = kwargs.get('num_epochs', 100)

        self.learning_rate = kwargs.get('learning_rate', None)
        self.num_epochs = kwargs.get('num_epochs', None)

        # optimizer = kwargs.get('optimizer', 'Rprop')

        optimizer = kwargs.get('optimizer', None)
        if optimizer == 'Rprop':
            self.optimizer = torch.optim.Rprop
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam
        elif optimizer == 'Adamw':
            self.optimizer = torch.optim.AdamW
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD
        elif optimizer == 'Adadelta':
            self.optimizer = torch.optim.Adadelta
        elif optimizer == 'Adagrad':
            self.optimizer = torch.optim.Adagrad
        elif optimizer == 'Adamax':
            self.optimizer = torch.optim.Adamax
        elif optimizer == 'ASGD':
            self.optimizer = torch.optim.ASGD
        elif optimizer == 'NAdam':
            self.optimizer = torch.optim.NAdam
        elif optimizer == 'RAdam':
            self.optimizer = torch.optim.RAdam
        elif optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop
        else:
            assert False



        # estimator = kwargs.get('estimator', 'MSE')
        estimator = kwargs.get('estimator', None)
        if estimator == 'MSE':
            self.estimator = nn.MSELoss
        else:
            assert False

        print(f'fitter config: {kwargs}')

    def fit(self, image_array):
        image = ImageDataset(image_array)
        # 参数配置

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 数据加载器
        train_loader = DataLoader(image, batch_size=len(image), shuffle=True)

        # 模型、损失函数、优化器
        model = deepcopy(self.model)
        model = model.to(device)
        # 交叉熵损失的计算包含了softmax，模型中不需要做softmax
        # loss = nn.CrossEntropyLoss()
        loss = self.estimator()
        # optimizer = self.optimizer(model.parameters(), lr=self.learning_rate,weight_decay=1e-6)   # weight_decay（默认值为 0）: 权重衰减，也称为 L2 正则化项。它用于控制参数的幅度，以防止过拟合。通常设置为一个小的正数。
        optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)


        # 训练
        loss_list = []


        for epoch in range(self.num_epochs):
            for i, (X_train, y_train) in enumerate(train_loader):
                X_train = X_train.to(device)
                pred = model(X_train)
                y_train = y_train.to(device)
                l = loss(pred, y_train)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()



            loss_list.append(l.item())



        print(f'loss is {l.item()}')
        # print(f"{image_path} has been fitted for {self.num_epochs} epochs, loss {l.item()}")

        # plt.plot(range(self.num_epochs), loss_list)
        # plt.xlabel("epoch")
        # plt.ylabel("loss")
        # plt.show()



        # # 使用matplotlib的面向对象API绘制损失曲线
        # self.fig3 = plt.figure(figsize=(8, 6))  # 您可以通过figsize参数调整图形的大小
        # self.ax3 = self.fig3.add_subplot(111)
        # self.ax3.plot(range(self.num_epochs), loss_list, label='Training Loss')  # 添加标签以便于创建图例
        # self.ax3.set_xlabel("Epoch")  # X轴标签
        # self.ax3.set_ylabel("Loss")  # Y轴标签
        # self.ax3.set_title("Loss Curve over Epochs")  # 图形标题
        # self.ax3.legend()  # 添加图例
        # self.fig3.canvas.draw()
        # plt.show()

        test_loader = DataLoader(image, batch_size=len(image), shuffle=False)
        with torch.no_grad():
            correct = 0
            total = 0
            for X_test, y_test in test_loader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                output = model(X_test)
                output = output.cpu().numpy()
                cloud = np.hstack((image.coordinates, output))
                if self.plot_manager is not None:
                    self.plot_manager.plot(data=image.scaled_cloud, model=cloud)

        parameter = model.parameters()
        feature = np.empty(0)
        for p in parameter:
            if p.requires_grad:
                f = p.detach().cpu().numpy().flatten()
                feature = np.append(feature, f)
        feature = feature[np.newaxis, :]
        return feature


'''

         torch.optim.Adamax(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
         其中各个参数的含义如下：

          params：包含模型参数的可迭代对象。
          lr：学习率，默认为0.001。
          betas：一个包含两个值的元组，分别为一阶矩和无穷范数的无穷范数的指数衰减率，默认为(0.9, 0.999)。
          eps：一个小的常数，用于提高数值稳定性，默认为1e-8。
          weight_decay：L2正则化项的权重衰减系数，默认为0，表示不使用权重衰减。

          weight_decay参数通常可以是大于等于0的任意实数，表示正则化项的权重系数。具体来说，weight_decay可以是以下几种不同数值：

                0：表示不使用权重衰减，即不添加正则化项到损失函数中。
                大于0小于1的小数：通常表示对于正则化项的贡献较小，有时称为L2范数约束（L2 weight decay）。
                大于1的整数：通常表示对于正则化项的贡献较大，有时称为权重正则化（weight regularization）。
                ∞：表示无穷大，逼近硬约束，有时称为权重衰减（weight decay）。

'''


## 不使用MLP网络!!消融实验：不使用网络训练
class Fitter_normal:
    def __init__(self):
        pass

    def fit_normal(self, image_array):
        train_data = ImageDataset_normal(image_array)

        return train_data.pixel_values





''' fitter for testiong: add nosie!!! '''

class Fitter_noise:
    def __init__(self, **kwargs):
        self.args = kwargs
        self.plot_manager = kwargs.get('plot_manager', None)

        model = kwargs.get('model',None)
        # model = kwargs.get('model', 'mlp')
        if model == 'mlp':
            from fit.models.mlp import MLPModel
            self.model = MLPModel()
        elif model == 'resnet':
            # TODO
            assert False
            from fit.models.resnet import ResNet  # TODO
            self.model = ResNet()
        else:
            assert False

        # self.learning_rate = kwargs.get('learning_rate', 0.0001)
        # self.num_epochs = kwargs.get('num_epochs', 100)

        self.learning_rate = kwargs.get('learning_rate', None)
        self.num_epochs = kwargs.get('num_epochs', None)

        # optimizer = kwargs.get('optimizer', 'Rprop')

        optimizer = kwargs.get('optimizer', None)
        if optimizer == 'Rprop':
            self.optimizer = torch.optim.Rprop
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam
        elif optimizer == 'Adamw':
            self.optimizer = torch.optim.AdamW
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD
        elif optimizer == 'Adadelta':
            self.optimizer = torch.optim.Adadelta
        elif optimizer == 'Adagrad':
            self.optimizer = torch.optim.Adagrad
        elif optimizer == 'Adamax':
            self.optimizer = torch.optim.Adamax
        elif optimizer == 'ASGD':
            self.optimizer = torch.optim.ASGD
        elif optimizer == 'NAdam':
            self.optimizer = torch.optim.NAdam
        elif optimizer == 'RAdam':
            self.optimizer = torch.optim.RAdam
        elif optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop
        else:
            assert False



        # estimator = kwargs.get('estimator', 'MSE')
        estimator = kwargs.get('estimator', None)
        if estimator == 'MSE':
            self.estimator = nn.MSELoss
        else:
            assert False

        print(f'fitter config: {kwargs}')

    def fit_noise(self, image_array):
        image = ImageDataset_add_noise(image_array)
        # 参数配置

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 数据加载器
        train_loader = DataLoader(image, batch_size=len(image), shuffle=True)

        # 模型、损失函数、优化器
        model = deepcopy(self.model)
        model = model.to(device)
        # 交叉熵损失的计算包含了softmax，模型中不需要做softmax
        # loss = nn.CrossEntropyLoss()
        loss = self.estimator()
        # optimizer = self.optimizer(model.parameters(), lr=self.learning_rate,weight_decay=1e-6)   # weight_decay（默认值为 0）: 权重衰减，也称为 L2 正则化项。它用于控制参数的幅度，以防止过拟合。通常设置为一个小的正数。
        optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)


        # 训练
        loss_list = []


        for epoch in range(self.num_epochs):
            for i, (X_train, y_train) in enumerate(train_loader):
                X_train = X_train.to(device)
                pred = model(X_train)
                y_train = y_train.to(device)
                l = loss(pred, y_train)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()



            loss_list.append(l.item())



        print(f'loss is {l.item()}')
        # print(f"{image_path} has been fitted for {self.num_epochs} epochs, loss {l.item()}")

        # plt.plot(range(self.num_epochs), loss_list)
        # plt.xlabel("epoch")
        # plt.ylabel("loss")
        # plt.show()



        # # 使用matplotlib的面向对象API绘制损失曲线
        # self.fig3 = plt.figure(figsize=(8, 6))  # 您可以通过figsize参数调整图形的大小
        # self.ax3 = self.fig3.add_subplot(111)
        # self.ax3.plot(range(self.num_epochs), loss_list, label='Training Loss')  # 添加标签以便于创建图例
        # self.ax3.set_xlabel("Epoch")  # X轴标签
        # self.ax3.set_ylabel("Loss")  # Y轴标签
        # self.ax3.set_title("Loss Curve over Epochs")  # 图形标题
        # self.ax3.legend()  # 添加图例
        # self.fig3.canvas.draw()
        # plt.show()

        test_loader = DataLoader(image, batch_size=len(image), shuffle=False)
        with torch.no_grad():
            correct = 0
            total = 0
            for X_test, y_test in test_loader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                output = model(X_test)
                output = output.cpu().numpy()
                cloud = np.hstack((image.coordinates, output))
                if self.plot_manager is not None:
                    self.plot_manager.plot(data=image.scaled_cloud, model=cloud)

        parameter = model.parameters()
        feature = np.empty(0)
        for p in parameter:
            if p.requires_grad:
                f = p.detach().cpu().numpy().flatten()
                feature = np.append(feature, f)
        feature = feature[np.newaxis, :]
        return feature


'''

         torch.optim.Adamax(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
         其中各个参数的含义如下：

          params：包含模型参数的可迭代对象。
          lr：学习率，默认为0.001。
          betas：一个包含两个值的元组，分别为一阶矩和无穷范数的无穷范数的指数衰减率，默认为(0.9, 0.999)。
          eps：一个小的常数，用于提高数值稳定性，默认为1e-8。
          weight_decay：L2正则化项的权重衰减系数，默认为0，表示不使用权重衰减。

          weight_decay参数通常可以是大于等于0的任意实数，表示正则化项的权重系数。具体来说，weight_decay可以是以下几种不同数值：

                0：表示不使用权重衰减，即不添加正则化项到损失函数中。
                大于0小于1的小数：通常表示对于正则化项的贡献较小，有时称为L2范数约束（L2 weight decay）。
                大于1的整数：通常表示对于正则化项的贡献较大，有时称为权重正则化（weight regularization）。
                ∞：表示无穷大，逼近硬约束，有时称为权重衰减（weight decay）。

'''


## 不使用MLP网络!!消融实验：不使用网络训练
class Fitter_normal_noise:
    def __init__(self):
        pass

    def fit_normal_noise(self, image_array):
        train_data = ImageDataset_normal_add_nosie(image_array)

        return train_data.pixel_values

