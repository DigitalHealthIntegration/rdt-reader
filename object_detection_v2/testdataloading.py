from utils import data_loader
import collections
import numpy as np
# data_loader.loadDataSeg("test")
# SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

# Spec = collections.namedtuple('Spec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])

# # the SSD orignal specs
# specs = [
#     Spec(38, 8, SSDBoxSizes(30, 60), [2]),
#     Spec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
#     Spec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
#     Spec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
#     Spec(3, 100, SSDBoxSizes(213, 264), [2]),
#     Spec(1, 300, SSDBoxSizes(264, 315), [2])
# ]


# x=data_loader.generateBoundingBox([0.5],[[[30,20]]],512)
# print(np.array(x))
# x,y,n = data_loader.loadDataObjSSD("test")
x,y,n = data_loader.loadDataObjSSDFromYoloFormat("train")
print(x.shape,y.shape)