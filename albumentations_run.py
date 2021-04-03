import albumentations as A
import cv2


# https://github.com/albumentations-team/albumentations
def invoke(num=100):
    image = cv2.imread("dataset/input/input.jpg")
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.ShiftScaleRotate(p=1),
    ])

    for x in range(num):
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        cv2.imwrite("./dataset/albumentations_output/transformed_{}.jpg".format(x), transformed_image)
