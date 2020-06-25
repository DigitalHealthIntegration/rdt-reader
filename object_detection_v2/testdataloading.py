from utils import data_loader
import collections
from core.config import cfg
import cv2
import numpy as np
import math
import ntpath

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
# # x,y,n = data_loader.loadDataObjSSD("test")
# x,y = data_loader.loadDataObjSSDFromYoloFormat("train")
# x,y= data_loader.loadDataObjSSDFromYoloFormat("test")

# print(x.shape,y.shape)
# filePathsTrain = data_loader.getImagePaths("train")
# labelDictTrain=data_loader.getAllAnnotations("train",filePathsTrain)
# filePathsTest = data_loader.getImagePaths("test")
# labelDictTest=data_loader.getAllAnnotations("test",filePathsTest)
filePathsTrain = data_loader.getImagePaths("train")
labelDictTrain=data_loader.getAllAnnotations("train",filePathsTrain)
filePathsTest = data_loader.getImagePaths("test")
labelDictTest=data_loader.getAllAnnotations("test",filePathsTest)
num_class        = cfg.TRAIN.NUMBER_CLASSES

trainGen = data_loader.image_generator(filePathsTrain,labelDictTrain,"train")
testGen = data_loader.image_generator(filePathsTest,labelDictTest,"test")
anchors = cfg.TRAIN.ANCHOR_ASPECTRATIO[0]
offset_for_class = len(anchors)*4
resize_dim=cfg.TEST.INPUT_SIZE
number_blocks = cfg.TRAIN.NUMBER_BLOCKS
resizefactor = [0,0]
resizefactor[0] = int(resize_dim[0]/number_blocks[0])
resizefactor[1] = int(resize_dim[1]/number_blocks[1])
featureMAPWH=[resize_dim[0]/number_blocks[0],resize_dim[1]/number_blocks[1]]

print("iterating")
for i in trainGen:
    for index,image in enumerate(i[0]):
        image = np.reshape(image,resize_dim)*255
        image = np.array(image,dtype=np.uint8)
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        label = i[1][index]
        print(image.shape)
        for row in range(label.shape[0]):
            for col in range(label.shape[1]):  
                for anch_id,anchor in enumerate(anchors):
                    ccol_offset,crow_offset,w_offset,h_offset = label[row,col,anch_id*4:(anch_id+1)*4]
                    class_preds = label[row,col,offset_for_class+(anch_id*num_class):offset_for_class+(anch_id+1)*num_class]
                    class_true = int(np.argmax(class_preds)/10)
                    color = [0,0,0]
                    # print(row,col,anch_id,x1,y1,x2,y2,class_true)
                    # fout.write(i[2][index]+" "+str(row)+" "+str(col)+" "+str(anch_id*4)+" "+str((anch_id+1)*4)+" "+str(np.argmax(class_preds))+"\n")
                    

                    if(class_true==3):
                        color[0]=255
                        color[1]=150
                        ccol = (col+0.5)*featureMAPWH[1]+ccol_offset*resize_dim[1]
                        crow =  (row+0.5)*featureMAPWH[0]+crow_offset*resize_dim[0]
                        w = anchor[1]*math.exp(w_offset)
                        h = anchor[0]*math.exp(h_offset)
                        col1,row1,col2,row2=data_loader.cxcy2xy([ccol,crow,w,h])
                        # x1,y1,x2,y2 = label[row,col,anch_id*4:(anch_id+1)*4]
                        image = cv2.rectangle(image,(int(col1),int(row1)),(int(col2),int(row2)),tuple(color),1)

                    elif max(class_preds)==0:
                        pass
                    else:
                        color[class_true]=255
                        ccol = (col+0.5)*featureMAPWH[1]+ccol_offset*resize_dim[1]
                        crow =  (row+0.5)*featureMAPWH[0]+crow_offset*resize_dim[0]
                        w = anchor[1]*math.exp(w_offset)
                        h = anchor[0]*math.exp(h_offset)
                        col1,row1,col2,row2=data_loader.cxcy2xy([ccol,crow,w,h])
                        # x1,y1,x2,y2 = label[row,col,anch_id*4:(anch_id+1)*4]
                        image = cv2.rectangle(image,(int(col1),int(row1)),(int(col2),int(row2)),tuple(color),1)

                        # print(class_true)
                    # targets[0,outer,inner,offset_for_class+(anch_id*num_class):offset_for_class+(anch_id+1)*num_class]=np.asarray(c)
                    # targets[0,outer,inner,anch_id*4:(anch_id+1)*4] = np.asarray([cx_offset,cy_offset,w_offset,h_offset])
        basename=ntpath.basename(i[2][index])
        cv2.imwrite("C:/Users/Kashyap/bkp/source/repos/rdt-reader/object_detection_v2/utils/data_check/"+basename+".jpg",image)
        