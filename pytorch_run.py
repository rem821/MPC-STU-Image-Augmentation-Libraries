import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
from skimage import io


# https://pytorch.org/vision/stable/transforms.html
class CustomDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
                Args:
                    csv_file (string): Path to the csv file with annotations.
                    root_dir (string): Directory with all the images.
                    transform (callable, optional): Optional transform to be applied
                        on a sample.
                """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = os.path.join(self.root_dir, self.landmarks_frame.iloc[index, 0])
        image = io.imread(img_path)
        # image = read_image(img_path)

        landmarks = self.landmarks_frame.iloc[index, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            image = self.transform(image)  # Here we should transform the whole sample (incl. landmarks) That would
            # mean to create custom Transforms that can work with dicts. We cannot call transform on image and
            # landmarks separately, as the transform function is random

        return image


def invoke(num=100):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(400),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    custom_dataset = CustomDataset(csv_file='dataset/annotation.csv', root_dir='dataset/input/', transform=transform)
    dataset_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=5, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataset_loader):

        # observe 4th batch and stop.
        if i_batch == 0:
            plt.figure()
            grid = utils.make_grid(sample_batched)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
