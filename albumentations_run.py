import albumentations as A
import cv2
import glob
import os
import time
from tqdm import tqdm
from benchmark_enum import BenchmarkType


# https://github.com/albumentations-team/albumentations
def invoke(num=100, benchmark_type=BenchmarkType.RESIZE) -> float:
    images = []
    for img in glob.glob("dataset/input/*.jpg"):
        n = cv2.imread(img)
        images.append(n)

    transform = setup_pipeline(benchmark_type)

    if not os.path.exists("./dataset/albumentations_output/"):
        os.mkdir("./dataset/albumentations_output/")

    start_time = time.time_ns()

    for x in tqdm(range(num)):
        transformed = transform(image=images[x % len(images)])
        transformed_image = transformed["image"]
        cv2.imwrite("./dataset/albumentations_output/transformed_{}.jpg".format(x), transformed_image)

    end_time = time.time_ns()
    milliseconds = (end_time - start_time) / 1_000_000
    # print("albumentations took {} milliseconds to run".format(milliseconds))
    return milliseconds


def setup_pipeline(benchmark_type) -> A.Compose:
    p = A.Compose
    if benchmark_type is BenchmarkType.HORIZONTAL_FLIP:
        p = A.Compose([
            A.HorizontalFlip(p=1)
        ])
    elif benchmark_type is BenchmarkType.VERTICAL_FLIP:
        p = A.Compose([
            A.VerticalFlip(p=1)
        ])
    elif benchmark_type is BenchmarkType.ROTATE:
        p = A.Compose([
            A.Rotate(p=1)
        ])
    elif benchmark_type is BenchmarkType.SHIFT_SCALE_ROTATE:
        p = A.Compose([
            A.ShiftScaleRotate(p=1)
        ])
    elif benchmark_type is BenchmarkType.BRIGHTNESS:
        p = A.Compose([
            A.RandomBrightness(p=1)
        ])
    elif benchmark_type is BenchmarkType.CONTRAST:
        p = A.Compose([
            A.RandomContrast(p=1)
        ])
    elif benchmark_type is BenchmarkType.RANDOM_CROP:
        p = A.Compose([
            A.RandomCrop(p=1, height=200, width=200)
        ])
    elif benchmark_type is BenchmarkType.RESIZE:
        p = A.Compose([
            A.Resize(p=1, width=400, height=400),
        ])
    elif benchmark_type is BenchmarkType.GRAYSCALE:
        p = A.Compose([
            A.ToGray(p=1),
        ])

    return p
