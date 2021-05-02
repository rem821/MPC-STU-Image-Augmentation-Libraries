import Augmentor
import os
import glob
import time
from benchmark_enum import BenchmarkType


# https://github.com/mdbloice/Augmentor
def invoke(num=100, benchmark_type=BenchmarkType.RESIZE, multithreaded=True) -> float:
    files = glob.glob('./dataset/augmentor_output/*')
    for f in files:
        os.remove(f)

    start_time = time.time_ns()

    p = Augmentor.Pipeline(source_directory="./dataset/input", output_directory="../augmentor_output")
    p = setup_pipeline(p, benchmark_type)
    p.sample(num, multi_threaded=multithreaded)

    end_time = time.time_ns()
    milliseconds = (end_time - start_time) / 1_000_000
    # print("Augmentor took {} milliseconds to run".format(milliseconds))
    return milliseconds


def setup_pipeline(p, benchmark_type) -> Augmentor.Pipeline:
    if benchmark_type is BenchmarkType.HORIZONTAL_FLIP:
        p.flip_left_right(probability=1)
    elif benchmark_type is BenchmarkType.VERTICAL_FLIP:
        p.flip_top_bottom(probability=1)
    elif benchmark_type is BenchmarkType.ROTATE:
        p.rotate(probability=1, max_right_rotation=25, max_left_rotation=25)
    elif benchmark_type is BenchmarkType.SHIFT_SCALE_ROTATE:
        p.crop_random(probability=1, percentage_area=0.8, randomise_percentage_area=True)
        p.rotate(probability=1, max_right_rotation=25, max_left_rotation=25)
        p.resize(probability=1, width=400, height=400)
    elif benchmark_type is BenchmarkType.BRIGHTNESS:
        p.random_brightness(probability=1, min_factor=0.2,max_factor=2)
    elif benchmark_type is BenchmarkType.CONTRAST:
        p.random_contrast(probability=1, min_factor=0.2, max_factor=2)
    elif benchmark_type is BenchmarkType.RANDOM_CROP:
        p.crop_random(probability=1, percentage_area=0.8, randomise_percentage_area=True)
    elif benchmark_type is BenchmarkType.RESIZE:
        p.resize(probability=1, width=400, height=400)
    elif benchmark_type is BenchmarkType.GRAYSCALE:
        p.greyscale(probability=1)

    return p
