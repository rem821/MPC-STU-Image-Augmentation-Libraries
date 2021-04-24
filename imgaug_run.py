from imgaug import augmenters as iaa
import cv2


# https://imgaug.readthedocs.io/en/latest/
def invoke(num=100):
    image = cv2.imread("dataset/input/0200.jpg")

    seq = iaa.Sequential([
        iaa.Snowflakes()
    ])

    for x in range(num):
        transformed_image = seq.augment_image(image=image)
        cv2.imwrite("./dataset/imgaug_output/transformed_{}.jpg".format(x), transformed_image)
