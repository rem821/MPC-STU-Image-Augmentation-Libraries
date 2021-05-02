from enum import Enum


class BenchmarkType(Enum):
    HORIZONTAL_FLIP = "Horizontal Flip"
    VERTICAL_FLIP = "Vertical Flip"
    ROTATE = "Rotate"
    SHIFT_SCALE_ROTATE = "Shift Scale Rotate"
    BRIGHTNESS = "Random Brightness"
    CONTRAST = "Random Contrast"
    RANDOM_CROP = "Random Crop"
    RESIZE = "Resize to 400x400 px"
    GRAYSCALE = "GrayScale"
