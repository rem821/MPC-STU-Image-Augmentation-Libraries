import augmentor_run as augmentor
import albumentations_run as albumentations
import imgaug_run as imgaug
import pytorch_run as pytorch
import autoaugment_run as autoaugment


augmentor.invoke(num=100)
albumentations.invoke(num=100)
imgaug.invoke(num=100)
pytorch.invoke(num=100)
autoaugment.invoke(num=100)
