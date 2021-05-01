import autoaugment
import PIL
import glob
import os
import time
from tqdm import tqdm


# AutoAugment: Learning Augmentation Policies from Data, Ekin D. Cubuk and Barret Zoph and Dandelion Mane and Vijay
# Vasudevan and Quoc V. Le, 2019
# https://github.com/DeepVoltaire/AutoAugment
def invoke(num=100):
    images = []
    for img in glob.glob("dataset/input/*.jpg"):
        n = PIL.Image.open(img)
        images.append(n)

    if not os.path.exists("./dataset/autoaugment_output/"):
        os.mkdir("./dataset/autoaugment_output/")

    start_time = time.time_ns()

    for x in tqdm(range(num)):
        policy = autoaugment.CIFAR10Policy()  # Also see CIFAR10Policy, SVHNPolicy
        transformed = policy(images[x % len(images)])
        transformed.save("dataset/autoaugment_output/{}.jpg".format(x))

    end_time = time.time_ns()
    print("autoaugment took {} milliseconds to run".format((end_time - start_time) / 1_000_000))
