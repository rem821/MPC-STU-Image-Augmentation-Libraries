import augmentor_run as augmentor
import albumentations_run as albumentations
import imgaug_run as imgaug
import pytorch_run as pytorch
import autoaugment_run as autoaugment
from benchmark_enum import BenchmarkType
import time
import csv

with open('benchmark.csv', 'w', newline='\n') as file:
    writer = csv.writer(file, delimiter='.')
    writer.writerow(["Action", "Augmentor", "Augmentor multithreaded", "Albumentations", "ImgAug", "PyTorch"])

    start_time = time.time_ns()
    for type in BenchmarkType:
        print("Benchmarking operation: {}".format(type.name))
        augmentor_points = augmentor.invoke(num=100, benchmark_type=type, multithreaded=False)
        augmentor_multi_points = augmentor.invoke(num=100, benchmark_type=type, multithreaded=True)
        albumentations_points = albumentations.invoke(num=100, benchmark_type=type)
        imgaug_points = imgaug.invoke(num=100, benchmark_type=type)
        pytorch_points = pytorch.invoke(num=100, benchmark_type=type)

        writer.writerow(
            [type.name, str(augmentor_points).replace('.', ','), str(augmentor_multi_points).replace('.', ','),
             str(albumentations_points).replace('.', ','),
             str(imgaug_points).replace('.', ','), str(pytorch_points).replace('.', ',')])

# autoaugment.invoke(num=100)

end_time = time.time_ns()
minutes = (end_time - start_time) / (1_000_000_000 * 60)
print("Benchmark ran for {} minutes".format(minutes))
