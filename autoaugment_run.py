import autoaugment
import PIL


# AutoAugment: Learning Augmentation Policies from Data, Ekin D. Cubuk and Barret Zoph and Dandelion Mane and Vijay
# Vasudevan and Quoc V. Le, 2019
# https://github.com/albumentations-team/albumentations
def invoke(num=100):
    for x in range(num):
        image = PIL.Image.open("dataset/input/0200.jpg")
        policy = autoaugment.CIFAR10Policy() #Also see CIFAR10Policy, SVHNPolicy
        transformed = policy(image)
        transformed.save("dataset/autoaugment_output/{}.jpg".format(x))
