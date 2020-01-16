import cv2
import numpy as np
import os
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug as ia
from core.config import cfg
from utils import utils
import json
from imgaug.augmentables.polys import Polygon
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import tensorflow as tf
import collections
import numpy as np
import itertools
import math
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import ntpath
import random
def cxcy2xy(bbox):
    return np.array([bbox[0]-bbox[2]/2,bbox[1]-bbox[3]/2,bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])

def xy2cxcy(bbox):
    xmin = min(bbox[0],bbox[2])
    xmax = max(bbox[0],bbox[2])
    ymin = min(bbox[1],bbox[3])
    ymax = max(bbox[1],bbox[3])
    
    return np.array([xmin+(xmax-xmin)/2,ymin+(ymax-ymin)/2,(xmax-xmin),(ymax-ymin)])

def loadDataObjSSDFromYoloFormat(dataset_type):
    """This function loads data from the directory of labels, it works with the yolo data format.
        
        Args:

            dataset_type (str) : train for loadinf the training set and test otherwise
        
        Returns:

            X,y,name

    """
    labelFileYoloFormat  = cfg.TRAIN.LABEL_FILE_YOLO if dataset_type == 'train' else cfg.TEST.LABEL_FILE_YOLO
    rootPathCentreLabel  = cfg.TRAIN.LABEL_PATH if dataset_type == 'train' else cfg.TEST.LABEL_PATH
    rootPathCroppedImages  = cfg.TRAIN.IMAGE_PATH if dataset_type == 'train' else cfg.TEST.IMAGE_PATH
    resize_dim = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
    resize_dim = tuple(resize_dim)
    batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
    data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG
    SeqAug    = returnAugmentationObj()   if dataset_type == 'train' else returnAugmentationObj()
    data_aug_upsample    = cfg.TRAIN.UPSAMPLE   if dataset_type == 'train' else cfg.TEST.UPSAMPLE
    rootPathForOutputCheckImages = cfg.TRAIN.OUTDATA
    classes          = utils.read_class_names(cfg.RDT_Reader.CLASSES)
    num_class        = cfg.TRAIN.NUMBER_CLASSES
    predictionScale  = cfg.TRAIN.PREDICTION_SCALE
    anchorAspect     = cfg.TRAIN.ANCHOR_ASPECTRATIO
    iou_thresh       = cfg.TRAIN.IOU_THRESH
    number_blocks = cfg.TRAIN.NUMBER_BLOCKS
    class_pos_id     = {}
    prime_numbers = [1,2,3,5,7,11,13,17]
    for cl in classes:
        # print(classes[cl],cl)
        class_pos_id[classes[cl]]=cl
    print(classes,class_pos_id)
    y=[]
    X=[]
    name=[]
    # labelNames = [int(classes[x]) for x in classes]
    # print(labelNames)
    all_annotations = {"frames":{}}
    image_list_id = {}
    with open(labelFileYoloFormat) as fin:
        files = fin.readlines()
        inpfil = [ntpath.basename(x.strip().split()[0]) for x in files]

    with open(labelFileYoloFormat) as fin:
        for line in fin:
            line=line.strip().split()
            basePath=ntpath.basename(line[0])
            try:
                all_annotations["frames"][basePath]=[]
                all_annotations["frames"][basePath+"path"]=line[0]
                for annots in line[1:]:
                    annots=annots.split(",")
                    tmpbox={"box": {"x1":float(annots[1]),"y1":float(annots[0]),"x2":float(annots[3]),"y2":float(annots[2])},"tags":[annots[-1]]}
                    all_annotations["frames"][basePath].append(tmpbox)
            except KeyError:
                pass
            except IndexError:
                print("LINE contenet:",line)
    # print(all_annotations)
    
    for ind,element in enumerate(inpfil):
        print(ind)
        if True: #"IMG_1514" in element:
            # print(all_annotations["frames"][element+"path"])
            try:
                #print(element)
                img_path = all_annotations["frames"][element+"path"]#os.path.join(rootPathCroppedImages,element)
                # img_path=img_path.replace("//","/").replace("/","\\").replace("object_detection_v2","object_detection_mobile_v2")
                img = cv2.imread(img_path,cv2.IMREAD_COLOR)
                original_size=img.shape
                ogimg = img
                img = cv2.pyrDown(img)
                img = cv2.pyrDown(img)
                # img = iaa.Fliplr(1.0)(images=img)
                # print(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img = cv2.resize(img,resize_dim)
                img = img[...,np.newaxis]
                # cnt = 1
                try:
                    objects = all_annotations["frames"][element]
                    # print(all_annotations[image_list_id[element]]["frames"][element])
                    targs = []
                    classES=[]
                    list_BBOX = []
                        # BoundingBox(x1=0.2*447, x2=0.85*447, y1=0.3*298, y2=0.95*298),
                    
                    for inde,obj in enumerate(objects):
                        c = np.zeros((num_class,))
                        if obj["tags"][0] in class_pos_id:
                            #print(obj["tags"][0])
                            c[int(obj["tags"][0])]=1
                            # print(obj["box"])
                            # print(original_size,resize_dim)
                            list_BBOX.append(BoundingBox(y1=obj["box"]["x1"], y2=obj["box"]["x2"], x1=obj["box"]["y1"], x2=obj["box"]["y2"]))
                            y1 = obj["box"]["x1"]/original_size[0]*resize_dim[0]
                            x1 = obj["box"]["y1"]/original_size[1]*resize_dim[1]
                            y2 = obj["box"]["x2"]/original_size[0]*resize_dim[0]
                            x2 = obj["box"]["y2"]/original_size[1]*resize_dim[1]
                            #list_BBOX.append(BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2))
                            targs.append([x1,y1,x2,y2])
                            classES.append(c)

                    priorBBoxes = generateBoundingBox(predictionScale,anchorAspect,resize_dim,number_blocks)
                    targets,match_ids=convert2target(priorBBoxes,targs,iou_thresh,classES,number_blocks,resize_dim,num_class,anchorAspect)
                    # print(classES)
                    # targets = np.reshape(targets,(1,15,15,5,7))
                    # for tar in targs:
                    #     x1,y1,x2,y2=tar
                    #     for ii,match_id in enumerate( match_ids):
                    #         # print(match_id)
                    #         # pred_x1=int(priorBBoxes[0,0,match_id[0],match_id[1],0]*resize_dim[0])

                    #         # pred_y1=int(priorBBoxes[0,0,match_id[0],match_id[1],1]*resize_dim[0])
                    #         # pred_x2=int(priorBBoxes[0,0,match_id[0],match_id[1],2]*resize_dim[0])
                    #         # pred_y2=int(priorBBoxes[0,0,match_id[0],match_id[1],3]*resize_dim[0])
                    #         # print(priorBBoxes.shape)
                    #         pcx,pcy,pw,ph =priorBBoxes[0,match_id[0],match_id[1],match_id[2],0],priorBBoxes[0,match_id[0],match_id[1],match_id[2],1],priorBBoxes[0,match_id[0],match_id[1],match_id[2],2],priorBBoxes[0,match_id[0],match_id[1],match_id[2],3] 
                            
                    #         pcx,pcy,pw,ph = xy2cxcy([pcx*resize_dim[0],pcy*resize_dim[0],pw*resize_dim[0],ph*resize_dim[0]])
                            
                    #         pcx,pcy,pw,ph = (pcx+targets[0,match_id[0],match_id[1],match_id[2],-4]),pcy+targets[0,match_id[0],match_id[1],match_id[2],-3],pw+targets[0,match_id[0],match_id[1],match_id[2],-2],ph+targets[0,match_id[0],match_id[1],match_id[2],-1]
                    #         # pcx,pcy,pw,ph=int(pcx*resize_dim[0]),int(pcy*resize_dim[0]),int(pw*resize_dim[0]),int(ph*resize_dim[0])
                    #         pred_x1,pred_y1,pred_x2,pred_y2 = cxcy2xy([pcx,pcy,pw,ph])
                    #         # print(pred_x1,pred_y1,pred_x2,pred_y2)
                    #         # pred_x1,pred_y1,pred_x2,pred_y2 = 
                    #         # print(pred_x1,pred_y1,pred_x2,pred_y2)
                    #         # print(targets[0,])
                    #         img = cv2.rectangle(img,(int(pred_x1),int(pred_y1)),(int(pred_x2),int(pred_y2)),tuple(targets[0,match_id[0],match_id[1],match_id[2],0:num_class]*255),1)

                    #     x1=int(x1)#*resize_dim[0])

                    #     y1=int(y1)#*resize_dim[0])
                    #     x2=int(x2)#*resize_dim[0])
                    #     y2=int(y2)#*resize_dim[0])


                    #     img = cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)                
                    #ogimg = img
                    img = img/255.0
                    X.append(img)
                    y.append(targets[0])
                    

                    if data_aug_upsample>1:
                        
                        for i in range(data_aug_upsample):
                            bbs = BoundingBoxesOnImage(list_BBOX, shape=ogimg.shape)
                            image_aug, bbs_aug = SeqAug(image=ogimg, bounding_boxes=bbs)
                            original_size=image_aug.shape
                            image_aug = cv2.pyrDown(image_aug)
                            image_aug = cv2.pyrDown(image_aug)
                            # img = iaa.Fliplr(1.0)(images=img)
                            # print(img)
                            image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2GRAY)
                            newtars = []
                            image_aug =image_aug[...,np.newaxis]
                            for tar in bbs_aug.bounding_boxes:
                                # print(tar)

                                #x1,y1,x2,y2=tar.x1,tar.y1,tar.x2,tar.y2
                                y1 = tar.y1/original_size[0]*resize_dim[0]
                                x1 = tar.x1/original_size[1]*resize_dim[1]
                                y2 = tar.y2/original_size[0]*resize_dim[0]
                                x2 = tar.x2/original_size[1]*resize_dim[1]
                                newtars.append([x1,y1,x2,y2])
                            # print("start")
                            targets,match_ids=convert2target(priorBBoxes,newtars,iou_thresh,classES,number_blocks,resize_dim,num_class,anchorAspect)
                            image_aug=image_aug/255.0
                            X.append(image_aug)
                            y.append(targets[0])
                    #         targets = np.reshape(targets,(1,number_blocks[0],number_blocks[1],len(anchorAspect[0]),num_class+4))

                    #         for tar in bbs_aug.bounding_boxes:
                    #             # print(tar)
                    #             x1,y1,x2,y2=tar.x1,tar.y1,tar.x2,tar.y2
                    #             for ii,match_id in enumerate( match_ids):
                    #                 # pred_x1=int(priorBBoxes[0,0,match_id[0],match_id[1],0]*resize_dim[0])

                    #                 # pred_y1=int(priorBBoxes[0,0,match_id[0],match_id[1],1]*resize_dim[0])
                    #                 # pred_x2=int(priorBBoxes[0,0,match_id[0],match_id[1],2]*resize_dim[0])
                    #                 # pred_y2=int(priorBBoxes[0,0,match_id[0],match_id[1],3]*resize_dim[0])
                    #                 # print(priorBBoxes.shape)
                    #                 pcx,pcy,pw,ph =priorBBoxes[0,match_id[0],match_id[1],match_id[2],0],priorBBoxes[0,match_id[0],match_id[1],match_id[2],1],priorBBoxes[0,match_id[0],match_id[1],match_id[2],2],priorBBoxes[0,match_id[0],match_id[1],match_id[2],3] 
                                    
                    #                 pcx,pcy,pw,ph = xy2cxcy([pcx*resize_dim[1],pcy*resize_dim[0],pw*resize_dim[1],ph*resize_dim[0]])
                                    
                    #                 pcx,pcy = (pcx+targets[0,match_id[0],match_id[1],match_id[2],-4]*resize_dim[1]),pcy+targets[0,match_id[0],match_id[1],match_id[2],-3]*resize_dim[0]
                    #                 pw=pw*math.exp(targets[0,match_id[0],match_id[1],match_id[2],-2])
                    #                 ph=ph*math.exp(targets[0,match_id[0],match_id[1],match_id[2],-1])
                    #                 # pcx,pcy,pw,ph=int(pcx*resize_dim[0]),int(pcy*resize_dim[0]),int(pw*resize_dim[0]),int(ph*resize_dim[0])
                    #                 pred_x1,pred_y1,pred_x2,pred_y2 = cxcy2xy([pcx,pcy,pw,ph])
                    #                 featclass=int(np.argmax(targets[0,match_id[0],match_id[1],match_id[2],0:num_class])/10)
                    #                 col = [0,0,0,0]
                    #                 col[featclass]=255
                    #                 # print(pred_x1,pred_y1,pred_x2,pred_y2)
                    #                 # pred_x1,pred_y1,pred_x2,pred_y2 = 
                    #                 # print(pred_x1,pred_y1,pred_x2,pred_y2)
                    #                 # print(targets[0,])
                    #                 # print("Writing",match_id,targets[0,match_id[0],match_id[1],match_id[2],0:num_class])
                    #                 if max(targets[0,match_id[0],match_id[1],match_id[2],0:num_class])==1:
                    #                     # print("Writing",match_id,targets[0,match_id[0],match_id[1],match_id[2],0:num_class])
                    #                     image_aug = cv2.rectangle(image_aug,(int(pred_x1),int(pred_y1)),(int(pred_x2),int(pred_y2)),tuple(col),1)

                    #             x1=int(x1)#*resize_dim[0])

                    #             y1=int(y1)#*resize_dim[0])
                    #             x2=int(x2)#*resize_dim[0])
                    #             y2=int(y2)#*resize_dim[0])
                                
                    #             image_aug = cv2.rectangle(image_aug,(x1,y1),(x2,y2),(255,0,255),1)      
                    #         cv2.imwrite(rootPathForOutputCheckImages+str(i)+obj["tags"][0]+element,image_aug)

                    # name.append(element)
                    #                 croppedImg = cv2.resize(croppedImg,resize_dim)
                    #                 cv2.imwrite(rootPathForOutputCheckImages+ str(i)+obj["tags"][0]+element,croppedImg)
                                # break
                    if ind==100 and dataset_type=="train":
                        break
                    elif ind==100 and dataset_type=="test":
                        break
                # break
                except KeyError:
                    print(image_list_id[element])
                    print("oop")

            except KeyError:
                print(element)
    X = np.array(X,dtype=np.float32)
    y = np.array(y,dtype=np.float32)

    return X,y,name


def generateBoundingBox(predictionScale,anchorAspect,shape,number_blocks):

    for ind,sc in enumerate(predictionScale):
        featureMAPWH=number_blocks
        
        def_bboxes=np.zeros((len(predictionScale),featureMAPWH[0],featureMAPWH[1],len(anchorAspect[ind]),4))
        
        for iii,anchs in enumerate(anchorAspect[ind]):
            for i in range(featureMAPWH[0]):
                for j in range(featureMAPWH[1]):
                    cx =  ((j+1)-0.5)/featureMAPWH[1]
                    cy = ((i+1)-0.5)/featureMAPWH[0]
                    w  = anchs[1]/shape[1]
                    h  = anchs[0]/shape[0]
                    def_bboxes[ind,i,j,iii]= cxcy2xy([cx,cy,w,h])
                    # print(def_bboxes[ind,i,j,iii])
    return def_bboxes

def convert2target(defaultbboxes,targs,thresh,targe_class,number_blocks,shape,num_class,anchorAspect):
    match_ids =[]
    scaledDownShape = number_blocks
    targets = np.zeros((1,scaledDownShape[0],scaledDownShape[1],len(anchorAspect[0]),(num_class+4)))
    cntpos=0
    cntneg=0
    for innnd,t in enumerate(targs) :
        x1y1x2y2 = t
        c = targe_class[innnd]
        #print(c)
        
        if t[0]<shape[1] and t[1]<shape[0]:
            if t[2]>shape[1]:
                t[2]=shape[1]
            if t[3]>shape[0]:
                t[3]=shape[0]
            for outer in range(scaledDownShape[0]):
                for inner in range(scaledDownShape[1]):
                    for anch_id in range(len(anchorAspect[0])):
                            # print(outer,inner,anch_id)
                            anch=defaultbboxes[0,outer,inner,anch_id]
                            # print("Before",anch)
                            anch_x1 = anch[0]*shape[1]
                            anch_y1 = anch[1]*shape[0]
                            anch_x2 = anch[2]*shape[1]
                            anch_y2 = anch[3]*shape[0] 
                            # print("After",anch)
                            # x1y1x2y2 = x1y1x2y2*shape
                            iou_score = iou([anch_x1,anch_y1,anch_x2,anch_y2],x1y1x2y2)
                            if max(c)==0:
                                print("FUNC")
                                #break
                            if iou_score>0.1:
                                cntpos+=1
                                cg_x,cg_y,g_w,g_h=xy2cxcy(x1y1x2y2)
                                cp_x,cp_y,p_w,p_h=xy2cxcy([anch_x1,anch_y1,anch_x2,anch_y2])
                                w_offset = math.log(g_w/p_w)
                                h_offset = math.log(g_h/p_h)
                                # print(c)
                                cx_offset,cy_offset = (cg_x-cp_x)/shape[1],(cg_y-cp_y)/shape[0]
                                #print("CX_offset",cx_offset,"CY_Offset",cy_offset,"w_offset",w_offset,"h_offset",h_offset,"Shape X Y",shape)
                                match_ids.append([outer,inner,anch_id])
                                targets[0,outer,inner,anch_id,0:num_class]=np.asarray(c)
                                targets[0,outer,inner,anch_id,num_class:] = np.asarray([cx_offset,cy_offset,w_offset,h_offset])
                            # print(targets[0,outer,inner,num_class:])
        else:
            pass    
    cntneg=cntpos/3
    if cntpos==0:
        cntneg=5
    while(cntneg>0):
        # print(cntneg)
        skip=0
        cc=np.zeros((num_class,))
        cc[-1]=1
        out=random.randint(0,number_blocks[0]-1)
        inner=random.randint(0,number_blocks[1]-1)
        anch_id=random.randint(0,len(anchorAspect[0])-1)
        anch=defaultbboxes[0,out,inner,anch_id]
        cx_offset,cy_offset,w_offset,h_offset=random.uniform(0.,1),random.uniform(0.01,1),random.uniform(-0.5,0.5),random.uniform(-0.5,0.5)
        anch_x1 = anch[0]*shape[1]
        anch_y1 = anch[1]*shape[0]
        anch_x2 = anch[2]*shape[1]
        anch_y2 = anch[3]*shape[0] 
        cp_x,cp_y,p_w,p_h=xy2cxcy([anch_x1,anch_y1,anch_x2,anch_y2])

        ca_x = cp_x+cx_offset*shape[1]
        ca_y = cp_y+cy_offset*shape[0]
        a_w = p_w*math.exp(w_offset)
        a_h = p_h*math.exp(h_offset)

        anch_x1,anch_y1,anch_x2,anch_y2=cxcy2xy([ca_x,ca_y,a_w,a_h])
        for innnd,t in enumerate(targs) :
            x1y1x2y2 = t
            iou_score = iou(x1y1x2y2,[anch_x1,anch_y1,anch_x2,anch_y2])
            # print(iou_score)
            if iou_score>0.:
                skip=1
                break
        if([out,inner,anch_id] not in match_ids) and (skip==0) and max(anch_x1,anch_x2)<shape[1] and max(anch_y1,anch_y2)<shape[0]:
            targets[0,out,inner,anch_id,0:num_class]=cc
            targets[0,out,inner,anch_id,num_class:]=np.asarray([cx_offset,cy_offset,w_offset,h_offset])
            cntneg-=1
            match_ids.append([out,inner,anch_id])
    targets = np.reshape(targets,(1,scaledDownShape[0]*scaledDownShape[1],len(anchorAspect[0]),(num_class+4)))
    return targets,match_ids

def iou(boxA,boxB):
    # print(boxA,boxB)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def loadDataObjSSD(dataset_type):
    """This function loads data from the directory of labels, it works with the yolo data format.
        
        Args:

            dataset_type (str) : train for loadinf the training set and test otherwise
        
        Returns:

            X,y,name

    """
    rootPathCentreLabel  = cfg.TRAIN.LABEL_PATH if dataset_type == 'train' else cfg.TEST.LABEL_PATH
    rootPathCroppedImages  = cfg.TRAIN.IMAGE_PATH if dataset_type == 'train' else cfg.TEST.IMAGE_PATH
    resize_dim = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
    resize_dim = tuple(resize_dim)
    batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
    data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG
    SeqAug    = returnAugmentationObj()   if dataset_type == 'train' else returnAugmentationObj(0)
    data_aug_upsample    = cfg.TRAIN.UPSAMPLE   if dataset_type == 'train' else cfg.TEST.UPSAMPLE
    rootPathForOutputCheckImages = cfg.TRAIN.OUTDATA
    classes          = utils.read_class_names(cfg.RDT_Reader.CLASSES)
    num_class        = cfg.TRAIN.NUMBER_CLASSES
    predictionScale  = cfg.TRAIN.PREDICTION_SCALE
    anchorAspect     = cfg.TRAIN.ANCHOR_ASPECTRATIO
    iou_thresh       = cfg.TRAIN.IOU_THRESH
    number_blocks = cfg.TRAIN.NUMBER_BLOCKS
    class_pos_id     = {}
    prime_numbers = [1,2,3,5,7,11,13,17]
    for cl in classes:
        # print(classes[cl],cl)
        class_pos_id[classes[cl]]=cl
    print(classes,class_pos_id)
    y=[]
    X=[]
    name=[]
    # labelNames = [int(classes[x]) for x in classes]
    # print(labelNames)
    all_annotations = []
    image_list_id = {}
    for ind,element in enumerate(os.listdir(rootPathCentreLabel)):
        with open(os.path.join(rootPathCentreLabel,element)) as fin:
            annotations = json.load(fin)
            all_annotations.append(annotations)
            for f in annotations["frames"].keys():
                image_list_id[f]=ind
    # print(image_list_id)
    for  ind,element in enumerate(os.listdir(rootPathCroppedImages)):
        print(ind)
        if True: #"IMG_1514" in element:
            img_path = os.path.join(rootPathCroppedImages,element)
            img = cv2.imread(img_path,cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            original_size=img.shape
            img = cv2.resize(img,resize_dim)
            img = img[...,np.newaxis]
            # cnt = 1
            try:
                objects = all_annotations[image_list_id[element]]["frames"][element]
                # print(all_annotations[image_list_id[element]]["frames"][element])
                targs = []
                classES=[]
                list_BBOX = []
                    # BoundingBox(x1=0.2*447, x2=0.85*447, y1=0.3*298, y2=0.95*298),

                for inde,obj in enumerate(objects):
                    c = np.zeros((num_class,))
                    if obj["tags"][0] in class_pos_id:
                        # print(obj["tags"][0])
                        if obj["tags"][0] in ["top","2"]:
                            c[0]=1
                        elif obj["tags"][0] in ["bottom","7"]:
                            c[1]=1
                        # print(obj["box"])
                        x1 = obj["box"]["x1"]/obj["width"]*resize_dim[0]
                        y1 = obj["box"]["y1"]/obj["height"]*resize_dim[0]
                        x2 = obj["box"]["x2"]/obj["width"]*resize_dim[0]
                        y2 = obj["box"]["y2"]/obj["height"]*resize_dim[0]
                        list_BBOX.append(BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2))
                        targs.append([x1,y1,x2,y2])
                        classES.append(c)

                priorBBoxes = generateBoundingBox(predictionScale,anchorAspect,resize_dim[0],number_blocks)
                targets,match_ids=convert2target(priorBBoxes,targs,iou_thresh,classES,number_blocks,resize_dim[0],num_class,anchorAspect)
                # print(classES)
                # targets = np.reshape(targets,(1,15,15,5,7))
                # for tar in targs:
                #     x1,y1,x2,y2=tar
                #     for ii,match_id in enumerate( match_ids):
                #         # print(match_id)
                #         # pred_x1=int(priorBBoxes[0,0,match_id[0],match_id[1],0]*resize_dim[0])

                #         # pred_y1=int(priorBBoxes[0,0,match_id[0],match_id[1],1]*resize_dim[0])
                #         # pred_x2=int(priorBBoxes[0,0,match_id[0],match_id[1],2]*resize_dim[0])
                #         # pred_y2=int(priorBBoxes[0,0,match_id[0],match_id[1],3]*resize_dim[0])
                #         # print(priorBBoxes.shape)
                #         pcx,pcy,pw,ph =priorBBoxes[0,match_id[0],match_id[1],match_id[2],0],priorBBoxes[0,match_id[0],match_id[1],match_id[2],1],priorBBoxes[0,match_id[0],match_id[1],match_id[2],2],priorBBoxes[0,match_id[0],match_id[1],match_id[2],3] 
                        
                #         pcx,pcy,pw,ph = xy2cxcy([pcx*resize_dim[0],pcy*resize_dim[0],pw*resize_dim[0],ph*resize_dim[0]])
                        
                #         pcx,pcy,pw,ph = (pcx+targets[0,match_id[0],match_id[1],match_id[2],-4]),pcy+targets[0,match_id[0],match_id[1],match_id[2],-3],pw+targets[0,match_id[0],match_id[1],match_id[2],-2],ph+targets[0,match_id[0],match_id[1],match_id[2],-1]
                #         # pcx,pcy,pw,ph=int(pcx*resize_dim[0]),int(pcy*resize_dim[0]),int(pw*resize_dim[0]),int(ph*resize_dim[0])
                #         pred_x1,pred_y1,pred_x2,pred_y2 = cxcy2xy([pcx,pcy,pw,ph])
                #         # print(pred_x1,pred_y1,pred_x2,pred_y2)
                #         # pred_x1,pred_y1,pred_x2,pred_y2 = 
                #         # print(pred_x1,pred_y1,pred_x2,pred_y2)
                #         # print(targets[0,])
                #         img = cv2.rectangle(img,(int(pred_x1),int(pred_y1)),(int(pred_x2),int(pred_y2)),tuple(targets[0,match_id[0],match_id[1],match_id[2],0:num_class]*255),1)

                #     x1=int(x1)#*resize_dim[0])

                #     y1=int(y1)#*resize_dim[0])
                #     x2=int(x2)#*resize_dim[0])
                #     y2=int(y2)#*resize_dim[0])


                #     img = cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)                
                ogimg = img
                img = img/255.0
                X.append(img)
                y.append(targets[0])

                if data_aug_upsample >1:
                    for i in range(data_aug_upsample):
                        bbs = BoundingBoxesOnImage(list_BBOX, shape=ogimg.shape)
                        image_aug, bbs_aug = SeqAug(image=ogimg, bounding_boxes=bbs)
                        newtars = []
                        for tar in bbs_aug.bounding_boxes:
                            # print(tar)

                            x1,y1,x2,y2=tar.x1,tar.y1,tar.x2,tar.y2
                            newtars.append([x1,y1,x2,y2])
                        targets,match_ids=convert2target(priorBBoxes,newtars,iou_thresh,classES,number_blocks,resize_dim[0],num_class,anchorAspect)
                        image_aug=image_aug/255.0
                        X.append(image_aug)
                        y.append(targets[0])
                        # targets = np.reshape(targets,(1,number_blocks,number_blocks,5,7))

                        # for tar in bbs_aug.bounding_boxes:
                        #     # print(tar)
                        #     x1,y1,x2,y2=tar.x1,tar.y1,tar.x2,tar.y2
                        #     for ii,match_id in enumerate( match_ids):
                        #         # print(match_id)
                        #         # pred_x1=int(priorBBoxes[0,0,match_id[0],match_id[1],0]*resize_dim[0])

                        #         # pred_y1=int(priorBBoxes[0,0,match_id[0],match_id[1],1]*resize_dim[0])
                        #         # pred_x2=int(priorBBoxes[0,0,match_id[0],match_id[1],2]*resize_dim[0])
                        #         # pred_y2=int(priorBBoxes[0,0,match_id[0],match_id[1],3]*resize_dim[0])
                        #         # print(priorBBoxes.shape)
                        #         pcx,pcy,pw,ph =priorBBoxes[0,match_id[0],match_id[1],match_id[2],0],priorBBoxes[0,match_id[0],match_id[1],match_id[2],1],priorBBoxes[0,match_id[0],match_id[1],match_id[2],2],priorBBoxes[0,match_id[0],match_id[1],match_id[2],3] 
                                
                        #         pcx,pcy,pw,ph = xy2cxcy([pcx*resize_dim[0],pcy*resize_dim[0],pw*resize_dim[0],ph*resize_dim[0]])
                                
                        #         pcx,pcy,pw,ph = (pcx+targets[0,match_id[0],match_id[1],match_id[2],-4]),pcy+targets[0,match_id[0],match_id[1],match_id[2],-3],pw+targets[0,match_id[0],match_id[1],match_id[2],-2],ph+targets[0,match_id[0],match_id[1],match_id[2],-1]
                        #         # pcx,pcy,pw,ph=int(pcx*resize_dim[0]),int(pcy*resize_dim[0]),int(pw*resize_dim[0]),int(ph*resize_dim[0])
                        #         pred_x1,pred_y1,pred_x2,pred_y2 = cxcy2xy([pcx,pcy,pw,ph])
                        #         # print(pred_x1,pred_y1,pred_x2,pred_y2)
                        #         # pred_x1,pred_y1,pred_x2,pred_y2 = 
                        #         # print(pred_x1,pred_y1,pred_x2,pred_y2)
                        #         # print(targets[0,])
                        #         if max(targets[0,match_id[0],match_id[1],match_id[2],0:num_class])==1:
                        #             image_aug = cv2.rectangle(image_aug,(int(pred_x1),int(pred_y1)),(int(pred_x2),int(pred_y2)),tuple(targets[0,match_id[0],match_id[1],match_id[2],0:num_class]*255),1)

                        #     x1=int(x1)#*resize_dim[0])

                        #     y1=int(y1)#*resize_dim[0])
                        #     x2=int(x2)#*resize_dim[0])
                        #     y2=int(y2)#*resize_dim[0])


                        #     image_aug = cv2.rectangle(image_aug,(x1,y1),(x2,y2),(255,255,0),2)      
                        # cv2.imwrite(rootPathForOutputCheckImages+element,image_aug)

                # name.append(element)
                                # croppedImg = cv2.resize(croppedImg,resize_dim)
                                # cv2.imwrite(rootPathForOutputCheckImages+ str(i)+obj["tags"][0]+element,croppedImg)
                            # break
                if ind==1 and dataset_type=="train":
                    break
                elif ind==1 and dataset_type=="test":
                    break
            # break
            except KeyError:
                print(image_list_id[element])
                print("oop")


    X = np.array(X,dtype=np.float32)
    y = np.array(y,dtype=np.float32)

    return X,y,name

def returnAugmentationObj(percentageOfChance=0.9):
    """This function returns an augementation pipeline which can be used to augment training data.
        
        Args:
            percentageOfChance (float) : Percentage of chance , eg: if it is 0.5, 50% of the images will go through the pipeline
        
        Returns:
            :class:`imgaug.augmenters.meta.Sequential` : Image augmentor 

    """

    sometimes = lambda aug: iaa.Sometimes(percentageOfChance, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    
    seq = iaa.Sequential(
        [
            
            sometimes(iaa.Affine(
                translate_percent={"x": (-0.03, 0.03),"y": (-0.03, 0.03)}, # translate by -x to +x percent (per axis)
                rotate=(-10, 10), # rotate by - to +x degrees
                scale=(0.5,1.25),
            )),
            # iaa.PerspectiveTransform(scale=(0.01, 0.016)) # Add perscpective transform

        ])
    return seq

def loadData(dataset_type):
    """This function loads data from the directory of labels, it works with the yolo data format.
        
        Args:

            dataset_type (str) : train for loadinf the training set and test otherwise
        
        Returns:

            X,y,name

    """
    rootPathCentreLabel  = cfg.TRAIN.LABEL_PATH if dataset_type == 'train' else cfg.TEST.LABEL_PATH
    rootPathCroppedImages  = cfg.TRAIN.IMAGE_PATH if dataset_type == 'train' else cfg.TEST.IMAGE_PATH
    resize_dim = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
    resize_dim = tuple(resize_dim)
    batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
    data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG
    SeqAug    = returnAugmentationObj()   if dataset_type == 'train' else returnAugmentationObj(0)
    data_aug_upsample    = cfg.TRAIN.UPSAMPLE   if dataset_type == 'train' else cfg.TEST.UPSAMPLE
    rootPathForOutputCheckImages = cfg.TRAIN.OUTDATA
    classes          = utils.read_class_names(cfg.RDT_Reader.CLASSES)
    num_class        = len(classes)
    
    y=[]
    X=[]
    name=[]
    labelNames = [int(classes[x]) for x in classes]
    # print(labelNames)
    for ind,element in enumerate(os.listdir(rootPathCentreLabel)):
        with open(os.path.join(rootPathCentreLabel,element)) as fin:
            imagep = os.path.join(rootPathCroppedImages,element.replace(".txt",".jpg"))
            # print(imagep)

            img = cv2.imread(os.path.join(rootPathCroppedImages,element.replace(".txt",".jpg")),cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            name.append(element)
            labels = fin.readlines()
            indices_labels=[]
            KPS=[]
            for x in labels:
                if int(x.strip().split()[0]) in labelNames:
                    indices_labels.append(int(x.strip().split()[0]))
            for x in labels:
                if int(x.strip().split()[0]) in labelNames:
                    KPS.append(Keypoint(x=int(float(x.strip().split()[1])*img.shape[1]),y=int(float(x.strip().split()[2])*img.shape[0])))
            
            # indices_labels=[int(x.strip().split()[0]) for x in labels]
            # KPS = [Keypoint(x=int(float(x.strip().split()[1])*img.shape[1]),y=int(float(x.strip().split()[2])*img.shape[0])) for x in labels]
            # labels = [(float(x.strip().split()[1]),float(x.strip().split()[2])) for x in labels]
            y_tmp = np.zeros((num_class,2))
            # print(element)
            for ind_aug in range(data_aug_upsample):
                kpsoi = KeypointsOnImage(KPS, shape=img.shape)
                images_aug_tr, keypoints_aug_tr = SeqAug(image=img,keypoints=kpsoi)  
                # images_aug_tr = cv2.resize(images_aug_tr, resize_dim, interpolation = cv2.INTER_CUBIC)
                images_aug_tr = cv2.cvtColor(images_aug_tr, cv2.COLOR_RGB2GRAY)

                edges = cv2.Canny(images_aug_tr,25,200,True)            
                # cv2.imshow("1",)    
                if False:
                    cv2.imwrite(rootPathForOutputCheckImages+"/edgedetected_"+str(ind)+str(ind_aug)+"_"+element.replace(".txt","")+".jpg",keypoints_aug_tr.draw_on_image(edges, size=10,color=255))
                    cv2.imwrite(rootPathForOutputCheckImages+"/augmented_"+str(ind)+str(ind_aug)+"_"+element.replace(".txt","")+".jpg",images_aug_tr)
                    
                # print(element)
                edges=cv2.resize(edges,resize_dim,interpolation = cv2.INTER_CUBIC)
                edges = np.reshape(edges,(edges.shape[0],edges.shape[1],1))
                # print(edges.shape) 
                X.append(edges)

                # labels = [[float(x.x)/img.shape[1],float(x.y)/img.shape[0]] for x in keypoints_aug_tr.keypoints]
                labels = [[float(x.x),float(x.y)] for x in keypoints_aug_tr.keypoints]

                # print (element,labels)
                for ii,index_of_lab in enumerate(indices_labels):
                    
                    y_tmp[ii][0]=labels[ii][0]
                    y_tmp[ii][1]=labels[ii][1]
                y.append(y_tmp)

                break
            # break

    X = np.array(X,dtype=np.float32)
    y = np.array(y,dtype=np.float32)

    return X,y,name

def loadDataSeg(dataset_type):
    """This function loads data from the directory of labels, it works with the yolo data format.
        
        Args:

            dataset_type (str) : train for loadinf the training set and test otherwise
        
        Returns:

            X,y,name

    """
    rootPathCentreLabel  = cfg.TRAIN.LABEL_PATH if dataset_type == 'train' else cfg.TEST.LABEL_PATH
    rootPathCroppedImages  = cfg.TRAIN.IMAGE_PATH if dataset_type == 'train' else cfg.TEST.IMAGE_PATH
    resize_dim = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
    resize_dim = tuple(resize_dim)
    batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
    data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG
    SeqAug    = returnAugmentationObj()   if dataset_type == 'train' else returnAugmentationObj(0)
    data_aug_upsample    = cfg.TRAIN.UPSAMPLE   if dataset_type == 'train' else cfg.TEST.UPSAMPLE
    rootPathForOutputCheckImages = cfg.TRAIN.OUTDATA
    classes          = utils.read_class_names(cfg.RDT_Reader.CLASSES)
    num_class        = len(classes)
    class_pos_id     = {}
    prime_numbers = [1,2,3,5,7,11,13,17]
    for cl in classes:
        # print(classes[cl],cl)
        class_pos_id[classes[cl]]=cl
    print(classes,class_pos_id)
    y=[]
    X=[]
    name=[]
    labelNames = [int(classes[x]) for x in classes]
    # print(labelNames)
    all_annotations = []
    image_list_id = {}
    for ind,element in enumerate(os.listdir(rootPathCentreLabel)):
        with open(os.path.join(rootPathCentreLabel,element)) as fin:
            annotations = json.load(fin)
            all_annotations.append(annotations)
            for f in annotations["frames"].keys():
                image_list_id[f]=ind
    # print(image_list_id)
    for  ind,element in enumerate(os.listdir(rootPathCroppedImages)):
        if True: #"IMG_1514" in element:
            img_path = os.path.join(rootPathCroppedImages,element)
            img = cv2.imread(img_path,cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_size=img.shape
            img = cv2.resize(img,resize_dim)
            # cnt = 1
            try:
                objects = all_annotations[image_list_id[element]]["frames"][element]
                # print(all_annotations[image_list_id[element]]["frames"][element])

                segmap = np.zeros((img.shape[0], img.shape[1], num_class+1), dtype=np.uint8)
                
                for inde,obj in enumerate(objects):
                    # print(obj["tags"])
                    correctedkey = obj["tags"][0]
                    if correctedkey in classes.values():
                        # print(correctedkey,element,class_pos_id[correctedkey]+1)
                        c = [0 for x in range(num_class+1)]                
                        c[class_pos_id[correctedkey]+1]=255
                        KPS = [Keypoint(x=points["x"]/original_size[1]*resize_dim[1],y=points["y"]/original_size[0]*resize_dim[0]) for points in obj["points"]]
                        kpsoi = KeypointsOnImage(KPS, shape=img.shape)
                        poly_ = Polygon(kpsoi.keypoints)
                        segmap = poly_.draw_on_image(
                                                        segmap,
                                                        color=tuple(c),
                                                        alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
                        # cnt+=1
                        # print(c)
                segmap = np.argmax(segmap, axis=-1)
                # segmap+=1
                segmap = segmap.astype(np.int32)
                # print("Shape:", segmap.shape, "min value:", segmap.min(), "max value:", segmap.max())
                segmap = SegmentationMapsOnImage(segmap, shape=img.shape)
                if False:
                    cv2.imwrite(os.path.join(rootPathForOutputCheckImages,"original"+element),segmap.draw_on_image(img)[0])
                
                for ind_aug in range(data_aug_upsample):
                    image_aug, segmap_aug = SeqAug(image=img, segmentation_maps=segmap)
                    if False:
                        cv2.imwrite(os.path.join(rootPathForOutputCheckImages,"aug"+element),segmap_aug.draw_on_image(image_aug)[0])

                    # image_aug = cv2.resize(image_aug,resize_dim)
                    segmap_small = segmap_aug#segmap_aug.resize((resize_dim[1],resize_dim[0]))
                    # image_aug = cv2.cvtColor(image_aug,cv2.COLOR_RGB2GRAY)

                    image_aug=image_aug/255.0
                    # image_aug = image_aug[...,np.newaxis]
                    # print(image_aug.shape,segmap_small.arr.shape)
                
                    image_aug = tf.keras.applications.mobilenet_v2.preprocess_input(image_aug)
                    X.append(image_aug)
                    # for iii in range(num_class):
                    #     segnew = np.where(segmap_small.arr==iii+1, prime_numbers[iii+1], segmap_small.arr) 
                    # target_tmp = np.reshape(segmap_small.arr,(resize_dim[0]*resize_dim[1],1))
                    target_tmp=[]
                    for i in range(num_class+1):
                        array = segmap_small.arr==i
                        
                        target_tmp.append(array.astype(int))
                    target_tmp = np.array(target_tmp)
                    target_tmp = np.reshape(target_tmp,(resize_dim[1],resize_dim[0],num_class+1))
                    # print(target_tmp.shape)
                    y.append(target_tmp)
                    # print(segnew.max(),segnew.min())
                    name.append(element)
                
                if ind==10:break
            # break
            except KeyError:
                print(image_list_id[element])
                print("oop")


    X = np.array(X,dtype=np.float32)
    y = np.array(y,dtype=np.float32)

    return X,y,name

def loadDataObj(dataset_type):
    """This function loads data from the directory of labels, it works with the yolo data format.
        
        Args:

            dataset_type (str) : train for loadinf the training set and test otherwise
        
        Returns:

            X,y,name

    """
    rootPathCentreLabel  = cfg.TRAIN.LABEL_PATH if dataset_type == 'train' else cfg.TEST.LABEL_PATH
    rootPathCroppedImages  = cfg.TRAIN.IMAGE_PATH if dataset_type == 'train' else cfg.TEST.IMAGE_PATH
    resize_dim = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
    resize_dim = tuple(resize_dim)
    batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
    data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG
    SeqAug    = returnAugmentationObj()   if dataset_type == 'train' else returnAugmentationObj(0)
    data_aug_upsample    = cfg.TRAIN.UPSAMPLE   if dataset_type == 'train' else cfg.TEST.UPSAMPLE
    rootPathForOutputCheckImages = cfg.TRAIN.OUTDATA
    classes          = utils.read_class_names(cfg.RDT_Reader.CLASSES)
    num_class        = len(classes)
    class_pos_id     = {}
    prime_numbers = [1,2,3,5,7,11,13,17]
    for cl in classes:
        # print(classes[cl],cl)
        class_pos_id[classes[cl]]=cl
    print(classes,class_pos_id)
    y=[]
    X=[]
    name=[]
    # labelNames = [int(classes[x]) for x in classes]
    # print(labelNames)
    all_annotations = []
    image_list_id = {}
    for ind,element in enumerate(os.listdir(rootPathCentreLabel)):
        with open(os.path.join(rootPathCentreLabel,element)) as fin:
            annotations = json.load(fin)
            all_annotations.append(annotations)
            for f in annotations["frames"].keys():
                image_list_id[f]=ind
    # print(image_list_id)
    for  ind,element in enumerate(os.listdir(rootPathCroppedImages)):
        if True: #"IMG_1514" in element:
            img_path = os.path.join(rootPathCroppedImages,element)
            img = cv2.imread(img_path,cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_size=img.shape
            img = cv2.resize(img,resize_dim)
            # cnt = 1
            try:
                objects = all_annotations[image_list_id[element]]["frames"][element]
                # print(all_annotations[image_list_id[element]]["frames"][element])
                
                for inde,obj in enumerate(objects):
                    c = np.zeros((3,))

                    if obj["tags"][0] in class_pos_id:
                        # print(obj["box"])
                        for i in range(4):
                             
                            if i%2==0:
                                l_x = obj["box"]["x1"]/original_size[1]*resize_dim[0]+i
                                u_x = obj["box"]["x2"]/original_size[1]*resize_dim[0]+i
                                l_y = obj["box"]["y1"]/original_size[0]*resize_dim[1]+i
                                u_y = obj["box"]["y2"]/original_size[0]*resize_dim[1]+i
                                c_x = (l_x+u_x)/2
                                c_y = (l_y+u_y)/2
                                l_x = c_x-65
                                u_x = c_x+65
                                l_y = c_y-25
                                u_y = c_y+25

                                negaexample_l_x = l_x-200
                                negaexample_u_x = u_x-200
                                negaexample_l_y = l_y-200
                                negaexample_u_y = u_y-200
                                
                                crop_neg = img[int(negaexample_l_y):int(negaexample_u_y),int(negaexample_l_x):int(negaexample_u_x),:]
                                # crop_neg = crop_neg/255.0
                                
                                ctmp = np.zeros((3,))
                                ctmp[-1]=1
                                if crop_neg.shape == (50,130,3):
                                    crop_neg = crop_neg/127.5 - 1
                                    crop_neg = cv2.resize(crop_neg,(160,160))
                                    X.append(crop_neg)
                                    y.append(ctmp)
                                    # 
                                    # cv2.imwrite(rootPathForOutputCheckImages+ str(i)+obj["tags"][0]+"neg"+element,crop_neg)
                            else:
                                l_x = obj["box"]["x1"]/original_size[1]*resize_dim[0]-i
                                u_x = obj["box"]["x2"]/original_size[1]*resize_dim[0]-i
                                l_y = obj["box"]["y1"]/original_size[0]*resize_dim[1]-i
                                u_y = obj["box"]["y2"]/original_size[0]*resize_dim[1]-i
                                c_x = (l_x+u_x)/2
                                c_y = (l_y+u_y)/2
                                l_x = c_x-65
                                u_x = c_x+65
                                l_y = c_y-25
                                u_y = c_y+25
                            # print(original_size,img.shape)
                            croppedImg = img[int(l_y):int(u_y),int(l_x):int(u_x),:]
                            # croppedImg = croppedImg/255.0
                            if obj["tags"][0] in ["top","2"]:
                                c[0]=1
                            else:
                                c[1]=1
                            if croppedImg.shape==(50,130,3):
                                croppedImg = croppedImg/127.5 -1
                                croppedImg = cv2.resize(croppedImg,(160,160))
                                X.append(croppedImg)
                                y.append(c)
                # name.append(element)
                                # croppedImg = cv2.resize(croppedImg,resize_dim)
                                # cv2.imwrite(rootPathForOutputCheckImages+ str(i)+obj["tags"][0]+element,croppedImg)
                            # break
                if ind==1:break
            # break
            except KeyError:
                print(image_list_id[element])
                print("oop")


    X = np.array(X,dtype=np.float32)
    y = np.array(y,dtype=np.float32)

    return X,y,name



if __name__ == "__main__":
    font = cv2.FONT_HERSHEY_SIMPLEX
