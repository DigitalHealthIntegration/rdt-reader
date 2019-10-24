import matplotlib; matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import cv2
from core.config import cfg
import matplotlib.pyplot as plt
from utils import data_loader
from core.model import ObjectDetection
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import time
# def load_graph(frozen_graph_filename):
#     # We load the protobuf file from the disk and parse it to retrieve the 
#     # unserialized graph_def
#     with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
#         graph_def = tf.compat.v1.GraphDef()
#         graph_def.ParseFromString(f.read())

#     # Then, we import the graph_def into a new Graph and returns it 
#     with tf.Graph().as_default() as graph:
#         # The name var will prefix every op/nodes in your graph
#         # Since we load everything in a new graph, this is not needed
#         tf.import_graph_def(graph_def, name="prefix")
#     return graph

cap = cv2.VideoCapture(0)
resize_dim =tuple(cfg.TEST.INPUT_SIZE) 
# testset            = data_loader.loadData('test')
# X,Y,names = testset[0],testset[1],testset[2]
interpreter = tf.lite.Interpreter(model_path="eval_model/tflite.lite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print(input_details)
# for x in X:
#     x = x[np.newaxis,...]
#     interpreter.set_tensor(input_details[0]['index'], x)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     print(output_data)



def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     frame = cv2.flip(frame,1)

#     # Our operations on the frame come here
#     img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
#     img=cv2.resize(img,(640,360),interpolation = cv2.INTER_CUBIC)

#     ogimg = img

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # edges = cv2.Canny(gray,25,200,True)            
#     # Display the resulting frame
#     img = img/127.5-1
#     print(img.shape)
#     output = []
#     st = time.time()

#     for i in range(1,9):
#         # subimages.append()    
#         for j in range(2,16):
#             inp = np.array(img[j*20:(j+1)*20,i*60:(i+1)*60],dtype=np.float32)
#             # inp = cv2.resize(inp,(160,160))
#             inp = np.reshape(inp,(1,inp.shape[0],inp.shape[1],1))
        
#             interpreter.set_tensor(input_details[0]['index'], inp)
#             interpreter.invoke()
#             # 
#             output_data = interpreter.get_tensor(output_details[0]['index'])

#             pred =np.argmax(output_data) 
#             conf = output_data[0][pred]
#             if pred == 0 :
#                 c=(0,0,255)
#             elif pred ==1:
#                 c=(255,0,0)
#             else:
#                 c=(255,255,255)

#             ogimg=cv2.rectangle(ogimg,(i*60,j*20),((i+1)*60,(j+1)*20),c,2)
#     et = time.time()
#     print("TIME TAKEN::::",(et-st))
#     # segmap = output_data>0.2
#     # segmap = np.argmax(segmap.astype(int), axis=-1)
#     # segmap = segmap.astype(np.int32)

#     # segmap = SegmentationMapsOnImage(segmap, shape=img.shape)
#     # outimg = segmap.draw_on_image(ogimg[0])[0]
#     # outimg = cv2.resize(ogimg,(512,512))
#     cv2.imshow('frame',ogimg)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# Load TFLite model and allocate tensors.
anchors=cfg.TRAIN.ANCHOR_ASPECTRATIO
number_blocks = cfg.TRAIN.NUMBER_BLOCKS
resizefactor = int(resize_dim[0]/number_blocks)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    fulziseimg =np.copy(img) 
    img=cv2.resize(img,(resize_dim[0],resize_dim[1]),interpolation = cv2.INTER_CUBIC)


    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray,25,200,True)            
    # Display the resulting frame
    img = img/255
    # print(img.shape)
    output = []
    # inp = cv2.resize(inp,(160,160))
    img = np.array(img,dtype=np.float32)
    inp = np.reshape(img,(1,img.shape[0],img.shape[1],1))
    st = time.time()
    box_0 =[]
    box_1 = []
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    # 
    et = time.time()

    preds = interpreter.get_tensor(output_details[0]['index'])
    preds = np.reshape(preds,(preds.shape[0],number_blocks,number_blocks,5,7))
    preds=preds[0]
    # print(np.where(preds[:,:,:,0:2]>0.8))
    axes = np.where(preds[:,:,:,0:2]>0.75)
    box_0 =[]
    box_1 = []
    for ind,ax_1 in enumerate(axes[0]):
        ax_2 = axes[1][ind]
        anch_id = axes[2][ind]
        tar_class = axes[-1][ind]
        offsets = preds[ax_1,ax_2,anch_id,2:]
        if tar_class==0:
            # print(row,col,inind,anch)

            cy = (ax_1+0.5)*resizefactor+offsets[-4]*img.shape[0]
            cx =  (ax_2+0.5)*resizefactor+offsets[-3]*img.shape[0]
            w = anchors[0][anch_id][0]+offsets[-2]*img.shape[0]
            h = anchors[0][anch_id][1]+offsets[-1]*img.shape[0]

            x1,y1,x2,y2=data_loader.cxcy2xy([cx,cy,w,h])
            box_0.append([x1,y1,x2,y2])
            # img = cv2.rectangle(img,(int(y1),int(x1)),(int(y2),int(x2)),255,1) 
        elif tar_class==1:
            # print(row,col,inind,anch)

            cy = (ax_1+0.5)*resizefactor+offsets[-4]*img.shape[0]
            cx =  (ax_2+0.5)*resizefactor+offsets[-3]*img.shape[0]
            h = anchors[0][anch_id][0]+offsets[-2]*img.shape[0]
            w = anchors[0][anch_id][1]+offsets[-1]*img.shape[0]

            x1,y1,x2,y2=data_loader.cxcy2xy([cx,cy,w,h])
            box_1.append([x1,y1,x2,y2])

 
    box_0 = np.array(box_0)
    box_1 = np.array(box_1)
    widthFactor = 1.0/256*640
    heightFactor = 1.0/256*480
    box_0=non_max_suppression_fast(box_0,0.9)
    box_1=non_max_suppression_fast(box_1,0.9)
    for b in box_0:
        fulziseimg = cv2.rectangle(fulziseimg,(int(b[0]*widthFactor),int(b[1]*heightFactor)),(int(b[2]*widthFactor),int(b[3]*heightFactor)),(255,0,0),1)
    for b in box_1:
        fulziseimg = cv2.rectangle(fulziseimg,(int(b[0]*widthFactor),int(b[1]*heightFactor)),(int(b[2]*widthFactor),int(b[3]*heightFactor)),(0,0,255),1)

    print("TIME TAKEN::::",(et-st))
    print(fulziseimg.shape)
    # ogimg = cv2.resize(ogimg,(512,512))
    cv2.imshow('frame',fulziseimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

