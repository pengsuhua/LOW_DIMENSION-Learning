from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# from utils.geometry import show_point_cloud



## 添加噪声!!!!
def add_gaussian_noise(image, mean=0., std=1.):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # 确保像素值在 [0, 255] 之间
    return noisy_image

def saltpepper_noise(image, proportion):
    image_copy = image.copy()
    img_Y, img_X = image.shape
    X = np.random.randint(img_X, size=(int(proportion * img_X * img_Y),))
    Y = np.random.randint(img_Y, size=(int(proportion * img_X * img_Y),))
    image_copy[Y, X] = np.random.choice([0.0, 1.0], size=(int(proportion * img_X * img_Y),))  ## 归一化后的图像 [0, 1] 之间，并确保噪声值在 [0, 1] 之间
    return image_copy


'''clean dataset for trian phase'''
class ImageDataset(Dataset):
    def __init__(self, image_array=None):
        Dataset.__init__(self)
        assert image_array is not None
        img = image_array
        indices = np.indices(img.shape)
        cloud = np.vstack((indices[0].flatten(), indices[1].flatten(), img.flatten())).transpose().astype('float32')
        self.cloud = np.copy(cloud)
        # self.coordinates = self.cloud[:, :2]/28.
        # self.pixel_values = self.cloud[:, -1]/255.
        self.coordinates = np.copy(cloud[:, :2])
        for i in range(2):
            self.coordinates[:, i] = cloud[:, i] / (img.shape[i] - 1)
        self.pixel_values = cloud[:, -1] / np.max(img)

        ###self.pixel_values = cloud[:, -1] / 255 * 0.99 + 0.01

        self.target = np.expand_dims(self.pixel_values, axis=1)
        self.scaled_cloud = np.hstack((self.coordinates, self.target))
        # show_point_cloud(cor)
        tmp = 0

    # def __init__(self, image_path=None):
    #     Dataset.__init__(self)
    #     assert image_path is not None
    #     img = Image.open(image_path)
    #     img = np.array(img)
    #     indices = np.indices(img.shape)
    #     self.cloud = np.vstack((indices[0].flatten(), indices[1].flatten(), img.flatten())).transpose().astype('float32')
    #     self.coordinates = self.cloud[:, :2]/28.
    #     self.pixel_values = self.cloud[:, -1]/255.
    #     self.target = np.expand_dims(self.pixel_values, axis=1)
    #     self.scaled_cloud = np.hstack((self.coordinates, self.target))
    #     # show_point_cloud(cor)
    #     tmp = 0

    def __len__(self):
        return self.cloud.shape[0]

    def __getitem__(self, item):
        return self.coordinates[item], self.target[item]




# 消融实验：不使用网络训练!!!!
class ImageDataset_normal(Dataset):
    def __init__(self, image_array=None):
        Dataset.__init__(self)
        assert image_array is not None
        img = image_array
        indices = np.indices(img.shape)
        cloud = np.vstack((indices[0].flatten(), indices[1].flatten(), img.flatten())).transpose().astype('float32')
        self.cloud = np.copy(cloud)
        # self.coordinates = self.cloud[:, :2]/28.
        # self.pixel_values = self.cloud[:, -1]/255.
        self.coordinates = np.copy(cloud[:, :2])
        for i in range(2):
            self.coordinates[:, i] = cloud[:, i] / (img.shape[i] - 1)

        self.pixel_values = cloud[:, -1] / np.max(img) ## init code!!!
        ### self.pixel_values = cloud[:, -1] / 255*0.99+0.01
        self.target = np.expand_dims(self.pixel_values, axis=1)
        self.scaled_cloud = np.hstack((self.coordinates, self.target))
        # show_point_cloud(cor)
        tmp = 0

    # def __init__(self, image_path=None):
    #     Dataset.__init__(self)
    #     assert image_path is not None
    #     img = Image.open(image_path)
    #     img = np.array(img)
    #     indices = np.indices(img.shape)
    #     self.cloud = np.vstack((indices[0].flatten(), indices[1].flatten(), img.flatten())).transpose().astype('float32')
    #     self.coordinates = self.cloud[:, :2]/28.
    #     self.pixel_values = self.cloud[:, -1]/255.
    #     self.target = np.expand_dims(self.pixel_values, axis=1)
    #     self.scaled_cloud = np.hstack((self.coordinates, self.target))
    #     # show_point_cloud(cor)
    #     tmp = 0

    def __len__(self):
        return self.cloud.shape[0]

    def __getitem__(self, item):
        return self.coordinates[item], self.target[item]





''' add noise to dataset for testing phase !!!!'''

class ImageDataset_add_noise(Dataset):
    def __init__(self, image_array=None):
        Dataset.__init__(self)
        assert image_array is not None
        img = image_array

        # # 假设你有一个函数来展示图像
        # plt.imshow(img, cmap='gray')
        # plt.show()
        ''' 添加噪声!!!! '''
        img = saltpepper_noise(img,0.2)
        # plt.imshow(img, cmap='gray')
        # plt.show()


        indices = np.indices(img.shape)
        cloud = np.vstack((indices[0].flatten(), indices[1].flatten(), img.flatten())).transpose().astype('float32')
        self.cloud = np.copy(cloud)
        # self.coordinates = self.cloud[:, :2]/28.
        # self.pixel_values = self.cloud[:, -1]/255.
        self.coordinates = np.copy(cloud[:, :2])
        for i in range(2):
            self.coordinates[:, i] = cloud[:, i] / (img.shape[i] - 1)
        self.pixel_values = cloud[:, -1] / np.max(img)

        ###self.pixel_values = cloud[:, -1] / 255 * 0.99 + 0.01

        self.target = np.expand_dims(self.pixel_values, axis=1)
        self.scaled_cloud = np.hstack((self.coordinates, self.target))
        # show_point_cloud(cor)
        tmp = 0

    # def __init__(self, image_path=None):
    #     Dataset.__init__(self)
    #     assert image_path is not None
    #     img = Image.open(image_path)
    #     img = np.array(img)
    #     indices = np.indices(img.shape)
    #     self.cloud = np.vstack((indices[0].flatten(), indices[1].flatten(), img.flatten())).transpose().astype('float32')
    #     self.coordinates = self.cloud[:, :2]/28.
    #     self.pixel_values = self.cloud[:, -1]/255.
    #     self.target = np.expand_dims(self.pixel_values, axis=1)
    #     self.scaled_cloud = np.hstack((self.coordinates, self.target))
    #     # show_point_cloud(cor)
    #     tmp = 0

    def __len__(self):
        return self.cloud.shape[0]

    def __getitem__(self, item):
        return self.coordinates[item], self.target[item]




# 消融实验：不使用网络训练!!!!
class ImageDataset_normal_add_nosie(Dataset):
    def __init__(self, image_array=None):
        Dataset.__init__(self)
        assert image_array is not None
        img = image_array

        ''' 添加噪声!!!! '''
        img = saltpepper_noise(img, 0.4)

        indices = np.indices(img.shape)
        cloud = np.vstack((indices[0].flatten(), indices[1].flatten(), img.flatten())).transpose().astype('float32')
        self.cloud = np.copy(cloud)
        # self.coordinates = self.cloud[:, :2]/28.
        # self.pixel_values = self.cloud[:, -1]/255.
        self.coordinates = np.copy(cloud[:, :2])
        for i in range(2):
            self.coordinates[:, i] = cloud[:, i] / (img.shape[i] - 1)

        self.pixel_values = cloud[:, -1] / np.max(img) ## init code!!!
        ### self.pixel_values = cloud[:, -1] / 255*0.99+0.01
        self.target = np.expand_dims(self.pixel_values, axis=1)
        self.scaled_cloud = np.hstack((self.coordinates, self.target))
        # show_point_cloud(cor)
        tmp = 0

    # def __init__(self, image_path=None):
    #     Dataset.__init__(self)
    #     assert image_path is not None
    #     img = Image.open(image_path)
    #     img = np.array(img)
    #     indices = np.indices(img.shape)
    #     self.cloud = np.vstack((indices[0].flatten(), indices[1].flatten(), img.flatten())).transpose().astype('float32')
    #     self.coordinates = self.cloud[:, :2]/28.
    #     self.pixel_values = self.cloud[:, -1]/255.
    #     self.target = np.expand_dims(self.pixel_values, axis=1)
    #     self.scaled_cloud = np.hstack((self.coordinates, self.target))
    #     # show_point_cloud(cor)
    #     tmp = 0

    def __len__(self):
        return self.cloud.shape[0]

    def __getitem__(self, item):
        return self.coordinates[item], self.target[item]
