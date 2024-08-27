from generalize.trainer import Trainer,Trainer_normal
from infer.predictor import Predictor,Predictor_normal
from utils.visualize.plot_manager import PlotManager
from fit.fitter import Fitter,Fitter_normal
import os
import time
import torch
import pathlib
os.chdir(pathlib.Path(__file__).parents[1])
import random
import numpy as np



# torch.manual_seed(100)
random.seed(0)  #设置python内置的随机数生成器的种子
np.random.seed(0) #设置Numpy的随机数生成器的种子
torch.manual_seed(0) #设置 PyTorch 的随机数生成器的种子。
torch.cuda.manual_seed_all(0) #设置Pytorch CUDA部分的随机数生成器的种子，确保使用GPU时的随机可复现性
torch.backends.cudnn.deterministic = True #禁用 cuDNN 的随机性，以确保在使用 cuDNN 加速时的结果可复现。
torch.backends.cudnn.benchmark = False #禁用 cuDNN 的自动寻找最优算法的功能，以确保结果的一致性。

config = dict(
    fitter=dict(
        model='mlp',
        # model='resnet',  # TODO
        optimizer='Adam',  # Adam,Adagrad
        estimator='MSE',
        learning_rate=0.0006, # 0.0005
        num_epochs=5000,  # 5000
    ),
    trainer=dict(
        dataset='MNIST',
        # dataset='fit/data/MNIST_mini/training',
        # training_data_path='fit/data/MNIST_mini/training',
        # training_data_path='fit/data/mnist/training',
        samples_per_class=120,                                                                                      # "samples_per_class=3"表示每个类别（或者标签）下有多少个样本。
    ),
    predictor=dict(
        dataset='MNIST',
        # dataset='fit/data/MNIST_mini/testing',
        # testing_data_path='fit/data/MNIST_mini/testing',
        # testing_data_path='fit/data/mnist/testing',
        samples_per_class=120,
    ),
    plot_manager=dict(
        # visualization='parallel',
        visualization='non-parallel',
    )
)


if __name__ == '__main__':
    plot_manager = PlotManager(**config['plot_manager'])  #  ** 时python中的一种语法，用于在函数调用或字典构造中进行解包操作。plot_manager = PlotManager(visualization='non-parallel')
    fitter = Fitter(**config['fitter'], plot_manager=plot_manager)

    print('\nstart training\n')
    trainer = Trainer(**config['trainer'], fitter=fitter)
    classifier = trainer.train()   # 现在使用的随机森林分类器

    print('\nstart testing\n')
    predictor = Predictor(**config['predictor'], classifier=classifier, fitter=fitter)
    predictor.predict()
    print(predictor.predicted)
    print(f'accuracy = {predictor.accuracy}')

    #time.sleep(360000)





''' 消融实验：不使用网络训练'''
# if __name__ == '__main__':
#     plot_manager = PlotManager(**config['plot_manager'])
#     fitter = Fitter_normal()
#
#     print('\nstart training\n')
#     trainer = Trainer_normal(**config['trainer'], fitter=fitter)
#     classifier = trainer.train_normal()   # 现在使用的随机森林分类器
#
#     print('\nstart testing\n')
#     predictor = Predictor_normal(**config['predictor'], classifier=classifier, fitter=fitter)
#     predictor.predict()
#     print(predictor.predicted)
#     print(f'accuracy = {predictor.accuracy}')
