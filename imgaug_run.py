from imgaug import augmenters as iaa
import cv2
import glob
import os
import time
from tqdm import tqdm
from benchmark_enum import BenchmarkType


# https://imgaug.readthedocs.io/en/latest/
def invoke(num=100, benchmark_type= BenchmarkType.RESIZE) -> float:
    images = []
    for img in glob.glob("dataset/input/*.jpg"):
        n = cv2.imread(img)
        images.append(n)

    seq = setup_pipeline(benchmark_type)

    if not os.path.exists("./dataset/imgaug_output/"):
        os.mkdir("./dataset/imgaug_output/")

    start_time = time.time_ns()
    for x in tqdm(range(num)):
        transformed_image = seq.augment_image(image=images[x % len(images)])
        cv2.imwrite("./dataset/imgaug_output/transformed_{}.jpg".format(x), transformed_image)

    end_time = time.time_ns()
    milliseconds = (end_time - start_time) / 1_000_000
    # print("imgaug took {} milliseconds to run".format(milliseconds))
    return milliseconds


def setup_pipeline(benchmark_type) -> iaa.Sequential:
    p = iaa.Sequential

    if benchmark_type is BenchmarkType.HORIZONTAL_FLIP:
        p = iaa.Sequential([
            iaa.Fliplr(p=1)
        ])
    elif benchmark_type is BenchmarkType.VERTICAL_FLIP:
        p = iaa.Sequential([
            iaa.Flipud(p=1)
        ])
    elif benchmark_type is BenchmarkType.ROTATE:
        p = iaa.Sequential([
            iaa.Rotate(rotate=(-25, 25)),
        ])
    elif benchmark_type is BenchmarkType.SHIFT_SCALE_ROTATE:
        p = iaa.Sequential([
            iaa.Rotate(rotate=(-25, 25)),
            iaa.Crop(px=(20, 100)),
            iaa.Resize(size=(400, 400))
        ])
    elif benchmark_type is BenchmarkType.BRIGHTNESS:
        p = iaa.Sequential([
            iaa.imgcorruptlike.Brightness(severity=3)
        ])
    elif benchmark_type is BenchmarkType.CONTRAST:
        p = iaa.Sequential([
            iaa.imgcorruptlike.Contrast(severity=3)
        ])
    elif benchmark_type is BenchmarkType.RANDOM_CROP:
        p = iaa.Sequential([
            iaa.Crop(px=(20, 100)),
        ])
    elif benchmark_type is BenchmarkType.RESIZE:
        p = iaa.Sequential([
            iaa.Resize(size=(400, 400)),
        ])
    elif benchmark_type is BenchmarkType.GRAYSCALE:
        p = iaa.Sequential([
            iaa.Grayscale()
        ])

    return p


