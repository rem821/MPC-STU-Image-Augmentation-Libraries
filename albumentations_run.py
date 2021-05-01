import albumentations as A
import cv2
import glob
import os
import time
from tqdm import tqdm


# https://github.com/albumentations-team/albumentations
def invoke(num=100):
    images = []
    for img in glob.glob("dataset/input/*.jpg"):
        n = cv2.imread(img)
        images.append(n)

    transform = A.Compose([
        A.ShiftScaleRotate(p=1),
    ])

    if not os.path.exists("./dataset/albumentations_output/"):
        os.mkdir("./dataset/albumentations_output/")

    start_time = time.time_ns()

    for x in tqdm(range(num)):
        transformed = transform(image=images[x % len(images)])
        transformed_image = transformed["image"]
        cv2.imwrite("./dataset/albumentations_output/transformed_{}.jpg".format(x), transformed_image)

    end_time = time.time_ns()
    print("albumentations took {} milliseconds to run".format((end_time - start_time) / 1_000_000))
