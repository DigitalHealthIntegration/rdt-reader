# -*- coding: utf-8 -*-
"""Class definitions for YOLO model and Red-Blue line detection model
"""
import sys
import os
from settings import RDT_GIT_ROOT
from settings import FLU_AUDERE_PATH
from settings import FLU_AUDERE_LINE_PATH
from settings import LINE_MODEL_VER
path = os.path.join(os.getcwd(),"tensorflow-yolov3")
sys.path.append(path)

path = os.path.join(RDT_GIT_ROOT,"tensorflow-yolov3")
sys.path.append(path)

jpg1Path = os.path.join(RDT_GIT_ROOT,"1.jpg")
jpg2Path = os.path.join(RDT_GIT_ROOT,"2.jpg")

#sys.path.append("C:\\Users\\developer\\Anaconda_3\\rdt-reader\\tensorflow-yolov3")
                
import json
import cv2 as cv
import numpy as np
from PIL import Image
from datetime import datetime as dt
from core import utils
from core.config import cfg

#grpc imports
import tensorflow as tf

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv.LUT(image, table)

def enhanceImage(img):
    img = np.uint8(img)
    newimg = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    clahe = cv.createCLAHE(10, (5,5))
    # newimg[1]=cv.normalize(newimg[1], 0, 255, cv.NORM_MINMAX)
    lab_planes = cv.split(newimg)
    lab_planes[1] = clahe.apply(lab_planes[1])
    lab = cv.merge(lab_planes)
    result=cv.cvtColor(lab, cv.COLOR_HLS2RGB)
    result = adjust_gamma(result,0.5)
    return result

def gaussBlur(img):
    img = cv.GaussianBlur(img,(1,11),0)
    return img

class YOLO:
    def __init__(self, input_size=512,weightsPath=FLU_AUDERE_PATH):
        """This function initializes the YOLO model and warms it up and returns predictor function handle
            
            Args:

                input_size (int) : Input size of image eg; (512x512x3)
                numClasses (int) : Number of type of objects being detected
                weightsPath (str) : Path to saved model
        
        """
        self.input_size = input_size
        self.num_classes =len(utils.read_class_names(cfg.YOLO.CLASSES)) 
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.weightsPath = weightsPath
        self.predictorFn=self.__hit_model()
        print("Exit Yolo constructor")


    def __hit_model(self):
        """This function tests the tf model and returns the predict function handler
        """
        #temp_inp_yolo = cv.imread("C:\\Users\\developer\\Anaconda_3\\rdt-reader\\1.jpg")
        temp_inp_yolo = cv.imread(jpg1Path)
        org_image = np.copy(temp_inp_yolo)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(temp_inp_yolo, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        
        try:
            temp_inp_yolo = np.array(image_data,dtype=np.float32)
            predict_fn = tf.contrib.predictor.from_saved_model(self.weightsPath)
            output_data = predict_fn({"input": temp_inp_yolo})
            return predict_fn
        except IOError:
            print("Unable to read either array or weight path")
            output_data = predict_fn({"input": temp_inp_yolo})
            return predict_fn
        


    def wrapper(self, image):
        """Wrapper method for the whole service for the object Detection service
            
            Args:

                image (numpy.ndarray) : BGR image

            Returns:
                
                list: Bounding boxes per category of class detected with confidence score
        """
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape
    
        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]



        start_grpc = dt.utcnow()

        result = self.predictorFn({"input":image_data})

        end_grpc = dt.utcnow()

        self.grpc_delta = end_grpc - start_grpc

        pred_sbbox = result["pred_sbbox"]
        pred_mbbox = result["pred_mbbox"]
        pred_lbbox = result["pred_lbbox"]
        # print("af pred",self.num_classes)
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        # bboxes = utils.nms(bboxes, self.iou_threshold)
        _image = utils.draw_bbox(org_image, bboxes, show_label=True)
        cv.imwrite("yolopred.jpg", _image)


        return bboxes 


class LineDetector:
    def __init__(self, input_size=[500,100],numClasses=4,weightsPath=FLU_AUDERE_LINE_PATH):
        """This function initializes the Line detector model and warms it up and returns predictor function handle
            
            Args:

                input_size (list) : Input size of image eg; [500,100] (500x100x3)
                numClasses (int) : Number of type of objects being detected
                weightsPath (str) : Path to saved model

        
        """ 
        self.input_size = input_size
        self.weightsPath = weightsPath
        self.num_classes = numClasses
        self.predictorFn=self.__hit_model()
    
    def renormalize(self,n, range1, range2):
        delta1 = range1[1] - range1[0]
        delta2 = range2[1] - range2[0]
        return (delta2 * (n - range1[0]) / delta1) + range2[0]

    def returnLOGker(self):
        sigma=48
        scaleConst1=1/(np.pi*(sigma**4))
        scaleConst2=1.0/(2*sigma**2)
        sizeofKernel=200
        sizeofKernel_2=int(sizeofKernel/2)
        LOG=np.zeros((sizeofKernel,sizeofKernel))
        for rows in range(sizeofKernel):
            i=rows-sizeofKernel_2
            for cols in range(sizeofKernel):
                j=cols-sizeofKernel_2
                LOG[rows,cols] = scaleConst1*(1-(i**2+j**2)*scaleConst2)*np.exp(-((i**2+j**2)*scaleConst2))
        LOG_oriented=np.zeros((int((sizeofKernel)/4),sizeofKernel))
        rowsubsample=0
        for rows in range(sizeofKernel):
            try:
                if rows%4==0:
                    for cols in range(sizeofKernel):
                        LOG_oriented[rowsubsample,cols]=LOG[rows,cols]
                    rowsubsample+=1
            except:
                pass
        LOG_oriented=LOG_oriented-np.mean(LOG_oriented)
        LOG_oriented=LOG_oriented*29000
        return LOG_oriented

    def LOG(self,im,kernel):
        img = im/255.0
        img = np.array(img,dtype=np.float32)
        imgYUV=cv.cvtColor(img,cv.COLOR_BGR2YUV)
        imgYUV[:,:,1:]=imgYUV[:,:,1:]-0.5
        filtered_img_GB=cv.filter2D(imgYUV , cv.CV_32F, kernel)*255
        return filtered_img_GB[:,:,1:]



    def gaborFilt(self,im):
        g_kernel = cv.getGaborKernel((9, 51), 6, np.pi/2, 0.2, 0.1, np.pi, ktype=cv.CV_32F)
        img = im/255.0
        img = np.array(img,dtype=np.float32)
        imgYUV=cv.cvtColor(img,cv.COLOR_BGR2YUV)

        filtered_img_GB = cv.filter2D(imgYUV*255, cv.CV_32F, g_kernel)
        renormalized=np.zeros((imgYUV.shape))
        # renormalized=renormalized[:,5:95,:]

        renormalized[:,:,0]=self.renormalize(filtered_img_GB[:,:,0],(np.min(filtered_img_GB[:,:,0]),np.max(filtered_img_GB[:,:,0])),(0,255))
        renormalized[:,:,1]=self.renormalize(filtered_img_GB[:,:,1],(np.min(filtered_img_GB[:,:,1]),np.max(filtered_img_GB[:,:,1])),(-128,127))
        renormalized[:,:,2]=self.renormalize(filtered_img_GB[:,:,2],(np.min(filtered_img_GB[:,:,2]),np.max(filtered_img_GB[:,:,2])),(-128,127))
        U_p=2.0
        V_p=13.0
        k1=U_p/V_p
        k2=1-k1
        newIMG=np.zeros((imgYUV.shape))
        # newIMG=newIMG[:,5:95,:]

        Final_U = k1*(renormalized[:,:,1]-renormalized[:,:,2])
        Final_V = k2*(renormalized[:,:,1]-renormalized[:,:,2])
        newIMG[:,:,1]=Final_U
        newIMG[:,:,2]=Final_V
        newIMG[:,:,0]=renormalized[:,:,0]

        newIMG[:,:,0]=self.renormalize(renormalized[:,:,0],(np.min(renormalized[:,:,0]),np.max(renormalized[:,:,0])),(0,255))
        newIMG[:,:,1]=self.renormalize(Final_U,(np.min(Final_U),np.max(Final_U)),(0,255))
        newIMG[:,:,2]=self.renormalize(Final_V,(np.min(Final_V),np.max(Final_V)),(0,255))
        im = newIMG[:,:,1:]
        return im


    def __hit_model(self):
        """This function tests the tf model and returns the predict function handler
        """

    
        #img = cv.imread("C:\\Users\\developer\\Anaconda_3\\rdt-reader\\2.jpg")
        img = cv.imread(jpg2Path)
        img=cv.resize(img,(100,2000))
        img = img[1500:1520,:,:]
        img = img/ 255.0
        img = np.array(img,dtype=np.float32)
        img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        if LINE_MODEL_VER==8:
            img = np.reshape(img[:,20:80,1:],(1,20,60,2))
        elif LINE_MODEL_VER>=6:
            img = np.reshape(img[:,:,1:],(1,20,100,2))
        else:
            img = np.reshape(img,(1,20,100,3))
        predict_fn = tf.contrib.predictor.from_saved_model(self.weightsPath)
        result = predict_fn({"input_image": img})
        return predict_fn


    def most_frequent(self,List): 
        return max(set(List), key = List.count) 
    
    def normalize(self,data):
        return (data - data.mean()) / data.std()
    
    def slidingMaxpool(self,truthLabels,windowSize,stride):
        
        noIterations = int(len(truthLabels)/windowSize + len(truthLabels)/stride)  
        y_averaged=[]
        for ind in range(noIterations):
            start = ind*stride
            end = start +windowSize
            tmp_data = truthLabels[start:end]
            if(len(tmp_data)==windowSize):
                m_tmp = self.most_frequent(tmp_data)
                y_averaged.append(m_tmp)
        return y_averaged
    
    def shredImage(self,img):
        y=[]
#         print("input shape",img.shape)
        if img.shape == (2000,100,3):
            pass
        else:
            img = cv.resize(img,(100,2000))
#             print(img.shape)
        img_inp = img[1000:1500,:,:]
        img_no_sclae = np.copy(img_inp)

        if LINE_MODEL_VER==1:
            img_inp = img_inp/255.0
            img_inp = np.array(img_inp,dtype=np.float32)
            img_inp = cv.cvtColor(img_inp,cv.COLOR_BGR2RGB)
        elif LINE_MODEL_VER==4: #Model 4, 5 ,6 ,7 ,8 
            img_inp = np.uint8(img_inp)
            img_inp = cv.cvtColor(img_inp,cv.COLOR_BGR2RGB)
            img_inp=gaussBlur(img_inp)
            img_inp= enhanceImage(img_inp)
            img_inp = img_inp/255.0
            img_inp = np.array(img_inp,dtype=np.float32)
        elif LINE_MODEL_VER==5: #Model 9 10 11 12
            print("MODEL TYPE ",LINE_MODEL_VER)
            img_inp = np.uint8(img_inp)
            img_inp = cv.cvtColor(img_inp,cv.COLOR_BGR2RGB)
            img_inp=gaussBlur(img_inp)
            img_inp = img_inp/255.0
            img_inp = np.array(img_inp,dtype=np.float32)
            img_inp = cv.cvtColor(img_inp,cv.COLOR_RGB2YCrCb)
        elif LINE_MODEL_VER==6: #Model 13
            img_inp= self.gaborFilt(img_inp)
            img_inp = img_inp/255.0
            img_inp = np.array(img_inp,dtype=np.float32)
        elif LINE_MODEL_VER==7:#Model 14 15
            LOGker = self.returnLOGker()
            img_inp = self.LOG(img_inp,LOGker)
            img_inp = img_inp/128.0
            img_inp = np.array(img_inp,dtype=np.float32)
        elif LINE_MODEL_VER==8:#Model 16 17
            img_inp=img_inp[:,20:80,:]
            LOGker = self.returnLOGker()
            img_inp =self.normalize(self.LOG(img_inp,LOGker)) 
            img_inp = img_inp/128.0
            img_inp = np.array(img_inp,dtype=np.float32)
        else:
            img_inp = img_inp/255.0
            img_inp = np.array(img_inp,dtype=np.float32)
            img_inp = cv.cvtColor(img_inp,cv.COLOR_BGR2YCrCb)

        for i in range(96):
            st = i*5
            end = st+20
            # print(end-st)
            shred_img = img_inp[st:end,:,:]
            shred_img_write = img_no_sclae[st:end,:,:]
            cv.imwrite("./shred/"+str(i)+".jpg",shred_img_write)
            shred_img = shred_img[np.newaxis]
            preds = self.predictorFn({"input_image": shred_img})
            preds=list(preds["predictions"][0])
            feat_class =preds.index(max(preds))
            if feat_class==0:
                y.append(0)
            elif feat_class==1:
                y.append(1)
            elif feat_class==2:
                y.append(2)
        print("sliding window predictions :",y)
        y_avg = self.slidingMaxpool(y,1,1)
        return y_avg

    def wrapper(self, image):
        """Wrapper method for the whole service for the Line Detection service
            
            Args:

                image (numpy.array) : BGR image

            Returns:
                
                list: Probability of detection
                list: Y-axis of predicted point
                numpy.ndarray: Image cropped
        """
        blue_detected = 0
        red_detected = 0
        virus_type = 0
        cntFromBlue=0
        cntFromRed=0
        predictions = self.shredImage(image)
        windowSize = 500/len(predictions)
        distanceR_B=[]
        for indexLoc,p in enumerate(predictions):
            if blue_detected==1:
                cntFromBlue+=1
            elif red_detected==1:
                cntFromRed+=1
            if p==2:
                pass
            elif p==1 and indexLoc<90:
                print("red",indexLoc)
                if red_detected==1 and blue_detected==0:
                        cntFromRed=0
                        virus_type=0
                elif blue_detected==0:
                    virus_type = 1
                    red_detected = 1
                else:
                    # print("here")
                    if red_detected==0 and cntFromBlue<20 and cntFromBlue>6:
                        virus_type = 2
                        red_detected=1
                        # if cntFromBlue>=20:
                        #     virus_type=0
                        #     cntFromBlue=0
                        #     blue_detected=0
                        #     red_detected=1
                        print("CENTER FROM BLUE",cntFromBlue)
                    elif cntFromBlue<20 and virus_type==1:
                        virus_type = 3
                    
                image = cv.circle(image,(50,int(indexLoc*windowSize+1010)),5, (0,0,255), 5)
            elif p==0 and indexLoc>28 and indexLoc<65:
                print("blue",indexLoc)
                if red_detected==1:
                    if cntFromRed<15 and (cntFromRed>3 or cntFromRed==0) :
                        virus_type=1
                        print("COUNT FROM RED",cntFromRed)
                    else:
                        virus_type=0
                        red_detected=0
                    
                    cntFromRed=0
                blue_detected=1
                image = cv.circle(image,(50,int(indexLoc*windowSize+1010)),5, (255,0,0), 5)
        return [image,virus_type,blue_detected]







