import albumentations as A
import cv2
import glob
from datetime import datetime


# https://github.com/albumentations-team/albumentations
def invoke(num=100):
    images = []
    for img in glob.glob("dataset/input/*.jpg"):
        n = cv2.imread(img)
        images.append(n)

    transform = A.Compose([
        A.ShiftScaleRotate(p=1),
    ])

    print("Albumentations start time:")
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    for x in range(num):
        print("{}/{}".format(x, num))
        transformed = transform(image=images[x % len(images)])
        transformed_image = transformed["image"]
        cv2.imwrite("./dataset/albumentations_output/transformed_{}.jpg".format(x), transformed_image)

    print("Albumentations end time:")
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])