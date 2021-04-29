from imgaug import augmenters as iaa
import cv2
import glob
from datetime import datetime


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

    print("Augmentor start time:")
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    for x in range(num):
        print("{}/{}".format(x, num))
        transformed_image = seq.augment_image(image=images[x % len(images)])
        cv2.imwrite("./dataset/imgaug_output/transformed_{}.jpg".format(x), transformed_image)

    print("Augmentor start time:")
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])



