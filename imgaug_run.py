from imgaug import augmenters as iaa
import cv2
import glob
import os
import time
from tqdm import tqdm


# https://imgaug.readthedocs.io/en/latest/
def invoke(num=100):
    images = []
    for img in glob.glob("dataset/input/*.jpg"):
        n = cv2.imread(img)
        images.append(n)

    seq = iaa.Sequential([
        iaa.Rot90(),
        iaa.PerspectiveTransform()
    ])

    if not os.path.exists("./dataset/imgaug_output/"):
        os.mkdir("./dataset/imgaug_output/")

    start_time = time.time_ns()
    for x in tqdm(range(num)):
        transformed_image = seq.augment_image(image=images[x % len(images)])
        cv2.imwrite("./dataset/imgaug_output/transformed_{}.jpg".format(x), transformed_image)

    end_time = time.time_ns()
    print("imgaug took {} milliseconds to run".format((end_time - start_time) / 1_000_000))



