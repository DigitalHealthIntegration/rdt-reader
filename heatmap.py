import cv2
import sys
import imgaug.augmenters as iaa

sys.path.insert(1,"D:\\source\\repos\\rdt-reader\\object_detection_v2")
import core.model as model
from core.config import cfg
import numpy as np
from utils import data_loader
inpImg="../object_detection_mobile_v2/train_hor_ratioCropped/I4.jpg"
import ntpath
import math
def rotate_bound(image, angle):
    """Return cropped RDT
        
        Args:

            image (numpy.ndarray) : Image with channels last format
            angle (numpy.float) : Angle to rotate image clockwise
            centers (list) : Centers of red and blue line (Used for debugging only)
        
        Returns:
            2-element tuple containing

                - **image** (*numpy.ndarray*): Rotated image 

                - **centers** (*list*): List of transformed centers (Used for debugging only)
                        
                 
    """  
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    tranformedCenters=[0,0,0,0,0,0]
    return cv2.warpAffine(image, M, (nW, nH)),tranformedCenters

def prepocessImageCOD(img,resize_dim):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = iaa.CropToFixedSize(320,180)(image=img)
    img = img[...,np.newaxis]
    img = img/255.0
    img = img[np.newaxis,...]
    return img


def main():
    num_class = cfg.TRAIN.NUMBER_CLASSES
    resize_dim=cfg.TEST.INPUT_SIZE
    anchors=cfg.TRAIN.ANCHOR_ASPECTRATIO
    number_blocks = cfg.TRAIN.NUMBER_BLOCKS
    resizefactor = [0,0]
    resizefactor[0] = int(resize_dim[0]/number_blocks[0])
    resizefactor[1] = int(resize_dim[1]/number_blocks[1])
    Model = model.ObjectDetection(True,"Model_KH_EXP/model_save.hdf5").model
    
    with open(cfg.TEST.LABEL_FILE_YOLO) as fin:
        for line in fin:
            imgpath=line.strip().split()[0]
            imgName = ntpath.basename(imgpath)
            fullsizeimg = cv2.imread(imgpath,cv2.IMREAD_COLOR)
            fullsizeimg,_ = rotate_bound(fullsizeimg,175)
            final_img = np.zeros((fullsizeimg.shape[0]*4+80,fullsizeimg.shape[1],3))        
            # print(img)
            final_img[0:fullsizeimg.shape[0],:fullsizeimg.shape[1],:]=fullsizeimg
            final_img[fullsizeimg.shape[0]+20:fullsizeimg.shape[0]*2+20,:fullsizeimg.shape[1],:]=fullsizeimg*0.2
            final_img[fullsizeimg.shape[0]*2+20:fullsizeimg.shape[0]*3+20,:fullsizeimg.shape[1],:]=fullsizeimg*0.2
            final_img[fullsizeimg.shape[0]*3+20:fullsizeimg.shape[0]*4+20,:fullsizeimg.shape[1],:]=fullsizeimg*0.2

            Input = prepocessImageCOD(fullsizeimg,resize_dim)
            # print(np.max(Input))
            predictions=Model.predict(Input)
            preds = np.reshape(predictions,(predictions.shape[0],number_blocks[0],number_blocks[1],4,8))
            preds=preds[0]
            box_0 =[]
            box_1 = []
            box_2 = []
            box_3 = []
            all_boxes = []
            for ax_1 in range(number_blocks[0]):
                for ax_2 in range(number_blocks[1]):
                    for anch_id in range(len(anchors[0])):
                        tar_class = np.argmax(preds[ax_1,ax_2,anch_id,0:num_class])
                        # print(tar_class)
                        prob=preds[ax_1,ax_2,anch_id,tar_class]
                        offsets = preds[ax_1,ax_2,anch_id,3:]
                        if tar_class==0:
                            # print(row,col,inind,anch)

                            cx = (ax_2+0.5)*resizefactor[1]+offsets[-4]*resize_dim[1]
                            cy =  (ax_1+0.5)*resizefactor[0]+offsets[-3]*resize_dim[0]
                            # w = anchors[0][anch_id][1]*math.exp(offsets[-2])
                            # h = anchors[0][anch_id][0]*math.exp(offsets[-1])
                            w = anchors[0][anch_id][1]+offsets[-2]*resize_dim[1]
                            h = anchors[0][anch_id][0]+offsets[-1]*resize_dim[0]

                            x1,y1,x2,y2=data_loader.cxcy2xy([cx,cy,w,h])
                            box_0.append([x1,y1,x2,y2,prob])
                            all_boxes.append([tar_class,x1,y1,x2,y2,prob])
                            # img = cv2.rectangle(img,(int(y1),int(x1)),(int(y2),int(x2)),255,1) 
                        elif tar_class==1:
                            # print(row,col,inind,anch)

                            cx = (ax_2+0.5)*resizefactor[1]+offsets[-4]*resize_dim[1]
                            cy =  (ax_1+0.5)*resizefactor[0]+offsets[-3]*resize_dim[0]
                            # w = anchors[0][anch_id][1]*math.exp(offsets[-2])
                            # h = anchors[0][anch_id][0]*math.exp(offsets[-1])
                            w = anchors[0][anch_id][1]+offsets[-2]*resize_dim[1]
                            h = anchors[0][anch_id][0]+offsets[-1]*resize_dim[0]

                            x1,y1,x2,y2=data_loader.cxcy2xy([cx,cy,w,h])
                            box_1.append([x1,y1,x2,y2,prob])
                            all_boxes.append([tar_class,x1,y1,x2,y2,prob])

                        elif tar_class==2:
                            # print(row,col,inind,anch)

                            cx = (ax_2+0.5)*resizefactor[1]+offsets[-4]*resize_dim[1]
                            cy =  (ax_1+0.5)*resizefactor[0]+offsets[-3]*resize_dim[0]
                            # w = anchors[0][anch_id][1]*math.exp(offsets[-2])
                            # h = anchors[0][anch_id][0]*math.exp(offsets[-1])
                            w = anchors[0][anch_id][1]+offsets[-2]*resize_dim[1]
                            h = anchors[0][anch_id][0]+offsets[-1]*resize_dim[0]

                            x1,y1,x2,y2=data_loader.cxcy2xy([cx,cy,w,h])
                            box_2.append([x1,y1,x2,y2,prob])
                            all_boxes.append([tar_class,x1,y1,x2,y2,prob])

                        elif tar_class==3:
                            # print(row,col,inind,anch)

                            cx = (ax_2+0.5)*resizefactor[1]+offsets[-4]*resize_dim[1]
                            cy =  (ax_1+0.5)*resizefactor[0]+offsets[-3]*resize_dim[0]
                            # w = anchors[0][anch_id][1]*math.exp(offsets[-2])
                            # h = anchors[0][anch_id][0]*math.exp(offsets[-1])
                            w = anchors[0][anch_id][1]+offsets[-2]*resize_dim[1]
                            h = anchors[0][anch_id][0]+offsets[-1]*resize_dim[0]

                            x1,y1,x2,y2=data_loader.cxcy2xy([cx,cy,w,h])
                            box_3.append([x1,y1,x2,y2,prob])

                        #     x1,y1,x2,y2=data_loader.cxcy2xy([cx,cy,w,h])
                        #     box_2.append([x1,y1,x2,y2,prob])
                        # elif tar_class==3:
                        #     print("oo")

            widthFactor = 1.0/resize_dim[1]*fullsizeimg.shape[1]
            heightFactor = 1.0/resize_dim[0]*fullsizeimg.shape[0]
            fullsizeimg=fullsizeimg*0.1

            box_0 = np.array(box_0,dtype=np.int32) 
            box_0=sorted(box_0,key=lambda x: x[3])   
            box_1 = np.array(box_1)
            box_1=sorted(box_1,key=lambda x: x[3])   

            box_2 = np.array(box_2)
            box_2=sorted(box_2,key=lambda x: x[3])   

            box_3=np.array(box_3)
            all_boxes=sorted(all_boxes,key=lambda x: x[3])
            print(fullsizeimg.shape)
            print(len(box_0),len(box_1),len(box_2))

            for b in all_boxes:
                tar=int(b[0])
                x1=int(b[1]*widthFactor)
                y1=int(b[2]*heightFactor)+(fullsizeimg.shape[0]*(tar+1)+20)
                x2=int(b[3]*widthFactor)
                y2=int(b[4]*heightFactor)+(fullsizeimg.shape[0]*(tar+1)+20)
                val=int(127*b[5]+128)
                final_img[y1:y2,x1:x2,tar]=val
                for i in range(3):
                    if i ==tar:
                        final_img[y1:y2,x1:x2,tar]=val
                    else:
                        final_img[y1:y2,x1:x2,i]=0

            cv2.imwrite("heatmap_175/"+imgName,final_img)
    # print(box_0,box_1)
main()





