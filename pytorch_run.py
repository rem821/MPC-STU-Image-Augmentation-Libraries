import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import cv2
import time
from tqdm import tqdm
from benchmark_enum import BenchmarkType


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


def invoke(num=100, benchmark_type=BenchmarkType.RESIZE) -> float:
    transform = setup_pipeline(benchmark_type)

    custom_dataset = CustomDataset(csv_file='dataset/annotation.csv', root_dir='dataset/input/', transform=transform)
    dataset_loader = torch.utils.data.DataLoader(custom_dataset, shuffle=True, num_workers=0)

    if not os.path.exists("./dataset/pytorch_output/"):
        os.mkdir("./dataset/pytorch_output/")

    start_time = time.time_ns()

    for x in tqdm(range(round(num / 5))):
        for idx, data in enumerate(dataset_loader):
            img = data[0].numpy()
            img = np.swapaxes(img, 0, 1)
            img = np.swapaxes(img, 1, 2)
            img = img * 255

            cv2.imwrite('dataset/pytorch_output/image_{}_{}.jpg'.format(x, idx), img)

    end_time = time.time_ns()
    milliseconds = (end_time - start_time) / 1_000_000
    # print("Pytorch took {} milliseconds to run".format(milliseconds))
    return milliseconds


def setup_pipeline(benchmark_type) -> transforms.Compose:
    p = transforms.Compose

    if benchmark_type is BenchmarkType.HORIZONTAL_FLIP:
        p = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor()])
    elif benchmark_type is BenchmarkType.VERTICAL_FLIP:
        p = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor()])
    elif benchmark_type is BenchmarkType.ROTATE:
        p = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=25),
            transforms.ToTensor()])
    elif benchmark_type is BenchmarkType.SHIFT_SCALE_ROTATE:
        p = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(400),
            transforms.ToTensor()])
    elif benchmark_type is BenchmarkType.BRIGHTNESS:
        p = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=1),
            transforms.ToTensor()])
    elif benchmark_type is BenchmarkType.CONTRAST:
        p = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(contrast=1),
            transforms.ToTensor()])
    elif benchmark_type is BenchmarkType.RANDOM_CROP:
        p = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(size=(100, 100)),
            transforms.ToTensor()])
    elif benchmark_type is BenchmarkType.RESIZE:
        p = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(400, 400)),
            transforms.ToTensor()])
    elif benchmark_type is BenchmarkType.GRAYSCALE:
        p = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor()])

    return p
