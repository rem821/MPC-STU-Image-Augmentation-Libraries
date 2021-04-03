import Augmentor
import os
import glob


# https://github.com/mdbloice/Augmentor
def invoke(num=100):
    files = glob.glob('./dataset/augmentor_output/*')
    for f in files:
        os.remove(f)

    p = Augmentor.Pipeline(source_directory="./dataset/input", output_directory="../augmentor_output")
    p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
    p.skew(probability=0.4, magnitude=0.8)
    p.histogram_equalisation(probability=0.3)
    p.random_distortion(probability=0.5, grid_width=10, grid_height=10, magnitude=10)
    p.crop_random(probability=0.2, percentage_area=0.5)
    p.random_color(probability=0.3, min_factor=0, max_factor=1)
    p.resize(probability=1, width=400, height=400)
    p.sample(num, multi_threaded=True)
