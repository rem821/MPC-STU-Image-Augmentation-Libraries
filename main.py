import augmentor_run as augmentor
import albumentations_run as albumentations
import imgaug_run as imgaug

# AutoAugment: Learning Augmentation Policies from Data, Ekin D. Cubuk and Barret Zoph and Dandelion Mane and Vijay Vasudevan and Quoc V. Le, 2019

augmentor.invoke(num=100)
albumentations.invoke(num=100)
imgaug.invoke(num=100)