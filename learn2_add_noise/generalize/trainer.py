import os
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder, MNIST
from torchvision.transforms import ToTensor, Normalize, Compose, Grayscale
from torch.utils.data import DataLoader, Subset
from fit.image_dataset import ImageDataset_normal
from tqdm import tqdm


''''--------------------------------------------------------------'''


''' init code'''
class Trainer:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.fitter = kwargs.get('fitter', None)
        assert self.fitter is not None
        dataset_name = kwargs.get('dataset', None)
        if os.path.exists(os.path.join(os.getcwd(), dataset_name)):
            transform_funcs = Compose([Grayscale(), ToTensor()])
            dataset = ImageFolder(root=dataset_name, transform=transform_funcs)
        elif dataset_name == "MNIST":
            transform_funcs = Compose([ToTensor()])
            # transform_funcs = Compose([
            #     ToTensor(),
            #     Normalize((0.1307,), (0.3081,))  # 标准化，手写数字数据集的通用参数
            # ])
            dataset = MNIST(root='./fit/data', train=True, download=True, transform=transform_funcs)
        else:
            assert False

        # subsets = {target: Subset(dataset, [i for i, (x, y) in enumerate(dataset) if y == target]) for _, target in dataset.class_to_idx.items()}
        # loaders = {target: DataLoader(subset, batch_size=64) for target, subset in subsets.items()}
        #
        # for target, loader in loaders.items():
        #     for i, (images, targets) in enumerate(loader):
        #         for j in range(images.shape[0]):
        #             image = images[j][0].numpy()
        #             print(f'fitting for the training image {i} {j}')
        #             feature = self.fitter.fit(image)
        #             self.features = feature if self.features is None else np.concatenate((self.features, feature),
        #                                                                                  axis=0)
        #             self.targets = targets[j] if self.targets is None else np.append(self.targets, targets[j])

        batch_size = 64

        x = dataset[0]
        self.features = None
        self.targets = None
        count = dict()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)   # "batch_size" 表示每次模型训练时所使用的数据批次（batch）的大小。换句话说，模型在每次更新参数时会同时处理多少个样本。

        for i, (images, targets) in enumerate(loader):
            for j in range(images.shape[0]): # images.shape[0]表示当前批次中的图片的数量。
                target = targets[j].item()
                count[target] = count.get(target, 0) + 1
                if min(count.values()) > self.config['samples_per_class']:
                    break
                if count[target] > self.config['samples_per_class']:
                    continue
                image = images[j][0].numpy()
                print(f'fitting for the training image of class {target}, sample {count[target]}')
                feature = self.fitter.fit(image)
                self.features = feature if self.features is None else np.concatenate((self.features, feature),
                                                                                     axis=0)
                self.targets = target if self.targets is None else np.append(self.targets, target)
            if min(count.values()) > self.config['samples_per_class']:
                break

        # training_path = kwargs.get('training_data_path', None)
        # categories = os.listdir(training_path)
        # self.data = {category: {} for category in categories}
        # self.features = None
        # self.targets = None
        # for category in categories:
        #     images = os.listdir(training_path + '/' + category)
        #     count = 0
        #     for image in images:
        #         image_path = training_path + '/' + category + '/' + image
        #         img = Image.open(image_path)
        #         img = np.array(img)
        #         # fitter = Fitter(plot_manager=self.plot_manager)
        #         print(f"fitting for the training image: {image_path}")
        #         feature = self.fitter.fit(img)
        #         self.features = feature if self.features is None else np.concatenate((self.features, feature), axis=0)
        #         self.targets = int(category) if self.targets is None else np.append(self.targets, int(category))
        #         self.data[category][image] = feature
        #         count += 1
        #         if count > self.config['samples_per_class']:
        #             break




    def train(self):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=90)
        clf.fit(self.features, self.targets)
        return clf




'''不归一化图像像素 '''

import torch
# class ToOriginalRangeTensor:
#     def __call__(self, pic):
#         # Convert a PIL Image or numpy.ndarray to tensor but keeping the original pixel range.
#         if isinstance(pic, Image.Image):
#             # Convert PIL Image to tensor
#             img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
#             img = img.view(pic.size[1], pic.size[0], 1)  # Grey image has one channel
#             img = img.transpose(0, 1).transpose(0, 2).contiguous()
#             return img.float()
#         raise TypeError(f'pic should be PIL Image. Got {type(pic)}')
#
# class Trainer:
#     def __init__(self, **kwargs):
#         self.config = kwargs
#         self.fitter = kwargs.get('fitter', None)
#         assert self.fitter is not None
#         dataset_name = kwargs.get('dataset', None)
#         if os.path.exists(os.path.join(os.getcwd(), dataset_name)):
#             # transform_funcs = Compose([Grayscale(), ToTensor()])
#             transform_funcs = Compose([Grayscale(), ToOriginalRangeTensor()])
#
#
#             dataset = ImageFolder(root=dataset_name, transform=transform_funcs)
#         elif dataset_name == "MNIST":
#             # transform_funcs = Compose([ToTensor()])
#
#             transform_funcs = Compose([ToOriginalRangeTensor()])
#
#             # transform_funcs = Compose([
#             #     ToTensor(),
#             #     Normalize((0.1307,), (0.3081,))  # 标准化，手写数字数据集的通用参数
#             # ])
#             dataset = MNIST(root='./fit/data', train=True, download=True, transform=transform_funcs)
#         else:
#             assert False
#
#         # subsets = {target: Subset(dataset, [i for i, (x, y) in enumerate(dataset) if y == target]) for _, target in dataset.class_to_idx.items()}
#         # loaders = {target: DataLoader(subset, batch_size=64) for target, subset in subsets.items()}
#         #
#         # for target, loader in loaders.items():
#         #     for i, (images, targets) in enumerate(loader):
#         #         for j in range(images.shape[0]):
#         #             image = images[j][0].numpy()
#         #             print(f'fitting for the training image {i} {j}')
#         #             feature = self.fitter.fit(image)
#         #             self.features = feature if self.features is None else np.concatenate((self.features, feature),
#         #                                                                                  axis=0)
#         #             self.targets = targets[j] if self.targets is None else np.append(self.targets, targets[j])
#
#         batch_size = 64
#
#         x = dataset[0]
#         self.features = None
#         self.targets = None
#         count = dict()
#         loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)   # "batch_size" 表示每次模型训练时所使用的数据批次（batch）的大小。换句话说，模型在每次更新参数时会同时处理多少个样本。
#
#         for i, (images, targets) in enumerate(loader):
#             for j in range(images.shape[0]): # images.shape[0]表示当前批次中的图片的数量。
#                 target = targets[j].item()
#                 count[target] = count.get(target, 0) + 1
#                 if min(count.values()) > self.config['samples_per_class']:
#                     break
#                 if count[target] > self.config['samples_per_class']:
#                     continue
#                 image = images[j][0].numpy()
#                 print(f'fitting for the training image of class {target}, sample {count[target]}')
#                 feature = self.fitter.fit(image)
#                 self.features = feature if self.features is None else np.concatenate((self.features, feature),
#                                                                                      axis=0)
#                 self.targets = target if self.targets is None else np.append(self.targets, target)
#             if min(count.values()) > self.config['samples_per_class']:
#                 break
#
#         # training_path = kwargs.get('training_data_path', None)
#         # categories = os.listdir(training_path)
#         # self.data = {category: {} for category in categories}
#         # self.features = None
#         # self.targets = None
#         # for category in categories:
#         #     images = os.listdir(training_path + '/' + category)
#         #     count = 0
#         #     for image in images:
#         #         image_path = training_path + '/' + category + '/' + image
#         #         img = Image.open(image_path)
#         #         img = np.array(img)
#         #         # fitter = Fitter(plot_manager=self.plot_manager)
#         #         print(f"fitting for the training image: {image_path}")
#         #         feature = self.fitter.fit(img)
#         #         self.features = feature if self.features is None else np.concatenate((self.features, feature), axis=0)
#         #         self.targets = int(category) if self.targets is None else np.append(self.targets, int(category))
#         #         self.data[category][image] = feature
#         #         count += 1
#         #         if count > self.config['samples_per_class']:
#         #             break
#
#
#
#
#     def train(self):
#         from sklearn.ensemble import RandomForestClassifier
#         clf = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=90)
#         clf.fit(self.features, self.targets)
#         return clf





# # ---------------------------------------------消融实验：不使用网络训练------------------------------------------------------------
'''

不归一化图像像素!!!!消融实验：不使用网络训练

'''
# import torch
# class ToOriginalRangeTensor:
#     def __call__(self, pic):
#         # Convert a PIL Image or numpy.ndarray to tensor but keeping the original pixel range.
#         if isinstance(pic, Image.Image):
#             # Convert PIL Image to tensor
#             img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
#             img = img.view(pic.size[1], pic.size[0], 1)  # Grey image has one channel
#             img = img.transpose(0, 1).transpose(0, 2).contiguous()
#             return img.float()
#         raise TypeError(f'pic should be PIL Image. Got {type(pic)}')
#
# class Trainer_normal:
#     def __init__(self, **kwargs):
#         self.config = kwargs
#         self.fitter = kwargs.get('fitter', None)
#         assert self.fitter is not None
#         dataset_name = kwargs.get('dataset', None)
#         if os.path.exists(os.path.join(os.getcwd(), dataset_name)):
#             # transform_funcs = Compose([Grayscale(), ToTensor()])
#             transform_funcs = Compose([Grayscale(), ToOriginalRangeTensor()])
#             dataset = ImageFolder(root=dataset_name, transform=transform_funcs)
#         elif dataset_name == "MNIST":
#             # transform_funcs = Compose([ToTensor()])
#             transform_funcs = Compose([ToOriginalRangeTensor()])
#             # transform_funcs = Compose([Grayscale()])
#             # transform_funcs = Compose([
#             #     ToTensor(),
#             #     Normalize((0.1307,), (0.3081,))  # 标准化，手写数字数据集的通用参数
#             # ])
#             dataset = MNIST(root='./fit/data', train=True, download=True, transform=transform_funcs)
#         else:
#             assert False
#
#         batch_size = 64
#
#         x = dataset[0]
#         self.features = None
#         self.targets = None
#         count = dict()
#         loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)  # "batch_size" 表示每次模型训练时所使用的数据批次（batch）的大小。换句话说，模型在每次更新参数时会同时处理多少个样本。
#
#         for i, (images, targets) in enumerate(loader):
#             for j in range(images.shape[0]):  # images.shape[0]表示当前批次中的图片的数量。
#                 target = targets[j].item()
#                 count[target] = count.get(target, 0) + 1
#                 if min(count.values()) > self.config['samples_per_class']:
#                     break
#                 if count[target] > self.config['samples_per_class']:
#                     continue
#                 image = images[j][0].numpy()
#                 print(f'fitting for the training image of class {target}, sample {count[target]}')
#                 feature = self.fitter.fit_normal(image)
#                 feature = feature[np.newaxis, :]    # #####
#                 self.features = feature if self.features is None else np.concatenate((self.features, feature),axis=0)
#
#                 self.targets = target if self.targets is None else np.append(self.targets,target)
#             if min(count.values()) > self.config['samples_per_class']:
#                 break
#
#
#
#
#     def train_normal(self):
#         from sklearn.ensemble import RandomForestClassifier
#         clf = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=90)
#         clf.fit(self.features, self.targets)
#
#         return clf


'''

归一化图像像素!!!! 消融实验：不使用网络训练

'''

class Trainer_normal:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.fitter = kwargs.get('fitter', None)
        assert self.fitter is not None
        dataset_name = kwargs.get('dataset', None)
        if os.path.exists(os.path.join(os.getcwd(), dataset_name)):
            transform_funcs = Compose([Grayscale(), ToTensor()])
            dataset = ImageFolder(root=dataset_name, transform=transform_funcs)
        elif dataset_name == "MNIST":
            transform_funcs = Compose([ToTensor()])
            # transform_funcs = Compose([
            #     ToTensor(),
            #     Normalize((0.1307,), (0.3081,))  # 标准化，手写数字数据集的通用参数
            # ])
            dataset = MNIST(root='./fit/data', train=True, download=True, transform=transform_funcs)
        else:
            assert False

        # subsets = {target: Subset(dataset, [i for i, (x, y) in enumerate(dataset) if y == target]) for _, target in dataset.class_to_idx.items()}
        # loaders = {target: DataLoader(subset, batch_size=64) for target, subset in subsets.items()}
        #
        # for target, loader in loaders.items():
        #     for i, (images, targets) in enumerate(loader):
        #         for j in range(images.shape[0]):
        #             image = images[j][0].numpy()
        #             print(f'fitting for the training image {i} {j}')
        #             feature = self.fitter.fit(image)
        #             self.features = feature if self.features is None else np.concatenate((self.features, feature),
        #                                                                                  axis=0)
        #             self.targets = targets[j] if self.targets is None else np.append(self.targets, targets[j])

        batch_size = 64

        x = dataset[0]
        self.features = None
        self.targets = None
        count = dict()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)   # "batch_size" 表示每次模型训练时所使用的数据批次（batch）的大小。换句话说，模型在每次更新参数时会同时处理多少个样本。

        for i, (images, targets) in enumerate(loader):
            for j in range(images.shape[0]): # images.shape[0]表示当前批次中的图片的数量。
                target = targets[j].item()
                count[target] = count.get(target, 0) + 1
                if min(count.values()) > self.config['samples_per_class']:
                    break
                if count[target] > self.config['samples_per_class']:
                    continue
                image = images[j][0].numpy()
                print(f'fitting for the training image of class {target}, sample {count[target]}')
                feature = self.fitter.fit_normal(image)   ##### !!!

                feature = feature[np.newaxis, :]

                self.features = feature if self.features is None else np.concatenate((self.features, feature),
                                                                                     axis=0)
                self.targets = target if self.targets is None else np.append(self.targets, target)
            if min(count.values()) > self.config['samples_per_class']:
                break

        # training_path = kwargs.get('training_data_path', None)
        # categories = os.listdir(training_path)
        # self.data = {category: {} for category in categories}
        # self.features = None
        # self.targets = None
        # for category in categories:
        #     images = os.listdir(training_path + '/' + category)
        #     count = 0
        #     for image in images:
        #         image_path = training_path + '/' + category + '/' + image
        #         img = Image.open(image_path)
        #         img = np.array(img)
        #         # fitter = Fitter(plot_manager=self.plot_manager)
        #         print(f"fitting for the training image: {image_path}")
        #         feature = self.fitter.fit(img)
        #         self.features = feature if self.features is None else np.concatenate((self.features, feature), axis=0)
        #         self.targets = int(category) if self.targets is None else np.append(self.targets, int(category))
        #         self.data[category][image] = feature
        #         count += 1
        #         if count > self.config['samples_per_class']:
        #             break




    def train_normal(self):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=90)
        clf.fit(self.features, self.targets)
        return clf


