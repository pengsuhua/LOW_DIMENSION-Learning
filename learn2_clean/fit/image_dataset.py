from torch.utils.data import Dataset
from PIL import Image
import numpy as np
# from utils.geometry import show_point_cloud

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
