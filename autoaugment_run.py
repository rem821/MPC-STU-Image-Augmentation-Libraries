import autoaugment
import PIL
import glob
from datetime import datetime


# AutoAugment: Learning Augmentation Policies from Data, Ekin D. Cubuk and Barret Zoph and Dandelion Mane and Vijay
# Vasudevan and Quoc V. Le, 2019
# https://github.com/albumentations-team/albumentations
def invoke(num=100):
    images = []
    for img in glob.glob("dataset/input/*.jpg"):
        n = PIL.Image.open(img)
        images.append(n)

    print("Autoaugment start time:")
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    for x in range(num):
        print("{}/{}".format(x, num))
        policy = autoaugment.CIFAR10Policy() #Also see CIFAR10Policy, SVHNPolicy
        transformed = policy(images[x % len(images)])
        transformed.save("dataset/autoaugment_output/{}.jpg".format(x))

    print("Autoaugment end time:")
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
