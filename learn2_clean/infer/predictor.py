import os
import numpy as np
from torchvision.datasets import ImageFolder, MNIST
from torchvision.transforms import ToTensor, Normalize, Compose, Grayscale
from torch.utils.data import DataLoader
from tqdm import tqdm


# import torch
# from PIL import Image
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

class Predictor:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.classifier = kwargs.get('classifier', None)
        # self.plot_manager = kwargs.get('plot_manager', None)
        self.fitter = kwargs.get('fitter', None)
        assert self.fitter is not None
        self.testing_path = kwargs.get('testing_data_path', None)
        self.dataset_name = kwargs.get('dataset', None)
        self.data = None
        self.features = None
        self.targets = None
        self.predicted = None
        self.correct = 0
        self.wrong = 0
        self.accuracy = -1

    def predict(self):
        if os.path.exists(os.path.join(os.getcwd(), self.dataset_name)):
           transform_funcs = Compose([Grayscale(), ToTensor()])

           ##transform_funcs = Compose([Grayscale(), ToOriginalRangeTensor()])

           dataset = ImageFolder(root=self.dataset_name, transform=transform_funcs)
        elif self.dataset_name == "MNIST":
            # transform_funcs = Compose([
            #     ToTensor(),
            #     Normalize((0.1307,), (0.3081,))  # 标准化，手写数字数据集的通用参数
            # ])

            transform_funcs = Compose([ToTensor()])

            ##transform_funcs = Compose([ToOriginalRangeTensor()])

            dataset = MNIST(root='./fit/data', train=False, download=True, transform=transform_funcs)
        else:
            assert False

        batch_size = 64
        count = dict()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for i, (images, targets) in enumerate(loader):
            for j in range(images.shape[0]):
                target = targets[j].item()
                count[target] = count.get(target, 0) + 1
                if min(count.values()) > self.config['samples_per_class']:
                    break
                if count[target] > self.config['samples_per_class']:
                    continue
                image = images[j][0].numpy()
                print(f'fitting for the testing image of class {target}, sample {count[target]}')
                feature = self.fitter.fit(image)
                predicted = self.classifier.predict(feature)
                if predicted == target:
                    self.correct += 1
                else:
                    self.wrong += 1
                self.features = feature if self.features is None else np.concatenate((self.features, feature),
                                                                                     axis=0)
                self.targets = target if self.targets is None else np.append(self.targets, target)
                self.predicted = predicted if self.predicted is None else np.append(self.predicted, predicted)
            if min(count.values()) > self.config['samples_per_class']:
                break
        self.accuracy = self.correct / (self.correct + self.wrong)

        # testing_path = self.testing_path
        # categories = os.listdir(testing_path)
        # self.data = {category: {} for category in categories}
        # for category in categories:
        #     images = os.listdir(testing_path + '/' + category)
        #     count = 0
        #     for image in images:
        #         image_path = testing_path + '/' + category + '/' + image
        #         # fitter = Fitter(plot_manager=self.plot_manager)
        #         feature = self.fitter.fit(image_path)
        #         predicted = self.classifier.predict(feature)
        #         if predicted == int(category):
        #             self.correct += 1
        #         else:
        #             self.wrong += 1
        #         self.features = feature if self.features is None else np.concatenate((self.features, feature), axis=0)
        #         self.targets = int(category) if self.targets is None else np.append(self.targets, int(category))
        #         self.predicted = predicted if self.predicted is None else np.append(self.predicted, predicted)
        #         self.data[category][image] = feature
        #         count += 1
        #         if count > self.config['samples_per_class']:
        #             break
        # self.accuracy = self.correct/(self.correct+self.wrong)


class Predictor_normal:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.classifier = kwargs.get('classifier', None)
        # self.plot_manager = kwargs.get('plot_manager', None)
        self.fitter = kwargs.get('fitter', None)
        assert self.fitter is not None
        self.testing_path = kwargs.get('testing_data_path', None)
        self.dataset_name = kwargs.get('dataset', None)
        self.data = None
        self.features = None
        self.targets = None
        self.predicted = None
        self.correct = 0
        self.wrong = 0
        self.accuracy = -1

    def predict(self):
        if os.path.exists(os.path.join(os.getcwd(), self.dataset_name)):
            transform_funcs = Compose([Grayscale(), ToTensor()])
            dataset = ImageFolder(root=self.dataset_name, transform=transform_funcs)
        elif self.dataset_name == "MNIST":
            # transform_funcs = Compose([
            #     ToTensor(),
            #     Normalize((0.1307,), (0.3081,))  # 标准化，手写数字数据集的通用参数
            # ])
            transform_funcs = Compose([ToTensor()])
            dataset = MNIST(root='./fit/data', train=False, download=True, transform=transform_funcs)
        else:
            assert False

        batch_size = 64
        count = dict()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for i, (images, targets) in enumerate(loader):
            for j in range(images.shape[0]):
                target = targets[j].item()
                count[target] = count.get(target, 0) + 1
                if min(count.values()) > self.config['samples_per_class']:
                    break
                if count[target] > self.config['samples_per_class']:
                    continue
                image = images[j][0].numpy()
                print(f'fitting for the testing image of class {target}, sample {count[target]}')
                feature = self.fitter.fit_normal(image)
                feature = feature[np.newaxis, :]
                predicted = self.classifier.predict(feature)
                if predicted == target:
                    self.correct += 1
                else:
                    self.wrong += 1

                feature = feature[np.newaxis, :]

                self.features = feature if self.features is None else np.concatenate((self.features, feature),
                                                                                     axis=0)
                self.targets = target if self.targets is None else np.append(self.targets, target)
                self.predicted = predicted if self.predicted is None else np.append(self.predicted, predicted)
            if min(count.values()) > self.config['samples_per_class']:
                break
        self.accuracy = self.correct / (self.correct + self.wrong)
