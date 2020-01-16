# -*- coding: utf-8 -*-
"""Flask server
"""
from tf_model_api import YOLO,LineDetector,gaussBlur,adjust_gamma,enhanceImage
import json
import os, io
import hashlib
import requests
from PIL import Image
from datetime import datetime as dt
import numpy as np
import cv2
import core.utils as utils
from PIL import Image
from base64 import decodestring
import re
import math
from requests_toolbelt import MultipartEncoder
import base64
import time
import os.path
from os import path
import settings as file_path_sets
# from tensorflow-yolov3.core import config as cfg
cannonicalArrow=[121.0,152.0,182.0]
cannonicalCpattern=[596.0,746.0,895.0]
cannonicalInfl=[699.0,874.0,1048.0]
ac_can = cannonicalCpattern[1]-cannonicalArrow[1]
ai_can = cannonicalInfl[1]-cannonicalArrow[1]
cannonicalA_C_Mid= np.array([449.0,30.0])
ref_A= np.array([cannonicalArrow[1]-cannonicalA_C_Mid[0],0.0])
ref_C= np.array([cannonicalCpattern[1]-cannonicalA_C_Mid[0],0.0])
ref_I= np.array([cannonicalInfl[1]-cannonicalA_C_Mid[0],0.0])
MAX_VALUE=100000.0

def reduceByConfidence(dictBoxC,dictBoxL):
    """This function handles multple object detection by selecting the one with the highest score.
        
        Args:

            dictBoxC (dict) : Objects detected and confidence
            dictBoxL (dict) : Objects detected and bounding box
            
        Returns:
       
            dict: Filtered list of Objects detected and bounding boxes 
    """
    filtDictBoxL={}
    for key in dictBoxC:
        maxConf=0
        try:
            if len(dictBoxC[key])>1:
                for ind,score in enumerate(dictBoxC[key]):
                    if(score>maxConf):
                        filtDictBoxL[key]=dictBoxL[key][ind]
                        maxConf=score
                    else:
                        pass
            else:
                filtDictBoxL[key]=dictBoxL[key][0]
        except Exception as e:
            print(e)
    return filtDictBoxL

def returnCentre(tlbr):
    """This function returns centre of bounding box.
        
        Args:

            tlbr (list) : list of values in str [topleft_x,topleft_y,bottomright_x,bottomright_y]
            
        Returns:
       
            list: Centre cordinates of bounding box 
    """
    int_tlbr = [int(tlbr[0]),int(tlbr[1])],[int(tlbr[2]),int(tlbr[3])]
    topleft,bottomright = [int(tlbr[0]),int(tlbr[1])],[int(tlbr[2]),int(tlbr[3])]
    centre=[0,0]
    centre[0] =int(topleft[0] + (bottomright[0]-topleft[0])/2) 
    centre[1] = int(topleft[1] + (bottomright[1]-topleft[1])/2) 
    return centre

def euclidianDistance(p1,p2):
    """Compute euclidian distance between p1 and p2
        
        Args:

            p1 (numpy.array) : X,Y of point 1
            p2 (numpy.array) : X,Y of point 2
        
            
        Returns:
       
            numpy.float: Distance between two points
    """
    return np.linalg.norm(p2-p1)

def angle_with_yaxis(p1,p2,img,centers,featsPres):
    """Compute angle by which image should be rotated,scale factor and returns a translated image
        
        Args:
            p1 (numpy.array) : X,Y of top pattern 1
            p2 (numpy.array) : X,Y of bottom arrow 2
            img (numpy.ndarray) : Image with channels last format
            centers (list) : Centers of red and blue line (Used for debugging only)
        Returns:
            3-element list containing
                    - **angle** (*numpy.float*): Angle to rotate Clock wise 
                    
                    - **image** (*numpy.ndarray*): Translated image 
                    - **centers** (*list*): List of transformed centers (Used for debugging only)
    """
    cent=[p1[0]+(p2[0]-p1[0])/2,p1[1]+(p2[1]-p1[1])/2]

    quad=1
    angleToRotateCW=0
    if(p2[0]==p1[0]):
        slope=0
    elif(p2[1]==p1[1]):
        p2[1]+=1
        slope = float(p2[1]-p1[1])/float(p2[0]-p1[0])
    else:
        slope = float(p2[1]-p1[1])/float(p2[0]-p1[0])

    ang=math.degrees(math.atan(slope))
    ydist=euclidianDistance(p1,p2)
    # Find qaudrant
    if (p1[0]-p2[0])<=0 and (p1[1]-p2[1])<=0:
        quad=1
    elif (p1[0]-p2[0])>0 and (p1[1]-p2[1])<0:
        quad=2
    elif (p1[0]-p2[0])>0 and (p1[1]-p2[1])>0:
        quad=3
    elif (p1[0]-p2[0])<0 and (p1[1]-p2[1])>0:
        quad=4
    # Compute angle to rotate
    if(ang>0 and quad==1):
        angleToRotateCW=90-ang
    elif(ang>0 and quad==3):
        angleToRotateCW=270-ang
    elif(ang<0 and quad==2):
        angleToRotateCW=360-(90+ang)
    elif(ang<0 and quad==4):
        ang=-1*ang
        angleToRotateCW=ang+90
    elif ang==0 and (p1[1]-p2[1])<0:
        angleToRotateCW=0
    elif ang==0 and (p1[1]-p2[1])>0:
        angleToRotateCW=180

    angleradian = (90-angleToRotateCW)*math.pi/180



    if featsPres==2:
        hyp =750
        cent[1]=cent[1]-hyp*math.sin(angleradian)
        cent[0]=cent[0]-hyp*math.cos(angleradian)
    elif featsPres==1:
        hyp = 155.0
        cent[1]=cent[1]+hyp*math.sin(angleradian)
        cent[0]=cent[0]+hyp*math.cos(angleradian)

    centimg=[img.shape[1]/2,img.shape[0]/2]
    transxRight=int(centimg[0]-cent[0])
    transyDown=int(centimg[1]-cent[1])
    tranformedCenters=[0,0,0,0,0,0]
    for ind,cents in enumerate(centers):
#         print(ind)
        cents[0] = cents[0]+transxRight
        cents[1] = cents[1]+transyDown
        tranformedCenters[2*ind]=cents[0]
        tranformedCenters[2*ind+1]=cents[1]
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([ [1,0,transxRight], [0,1,transyDown] ])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    if featsPres==0:
        fy=1220.0/ydist # scale computed from reference image
    elif featsPres==1:
        fy=900.0/ydist
    elif featsPres==2:
        fy=311.0/ydist
    return angleToRotateCW,img_translation,fy,quad,tranformedCenters

def angle_with_yaxis2(cp1,ap2,ip3,img,centers,featsPres):
    """Compute angle by which image should be rotated,scale factor and returns a translated image
        
        Args:

            p1 (numpy.array) : X,Y of top pattern 1
            p2 (numpy.array) : X,Y of bottom arrow 2
            img (numpy.ndarray) : Image with channels last format
            centers (list) : Centers of red and blue line (Used for debugging only)
        Returns:
            3-element list containing

                    - **angle** (*numpy.float*): Angle to rotate Clock wise 
                    
                    - **image** (*numpy.ndarray*): Translated image 

                    - **centers** (*list*): List of transformed centers (Used for debugging only)

    """
    ac_can_full = 1250
    ai_can_full = 1500
    cent=[cp1[0]+(ap2[0]-cp1[0])/2,cp1[1]+(ap2[1]-cp1[1])/2]

    quad=1
    angleToRotateCW=0
    # if(p2[0]==p1[0]):
    #     slope=0
    # elif(p2[1]==p1[1]):
    #     p2[1]+=1
    #     slope = float(p2[1]-p1[1])/float(p2[0]-p1[0])
    # else:
    #     slope = float(p2[1]-p1[1])/float(p2[0]-p1[0])
    th1=angleOfLine(ap2,cp1)
    th2=angleOfLine(ap2,ip3)
    theta=(th1+th2)/2
    theta=th1
    if(theta<0): theta+=2*math.pi
    #avoid feature orientations which are very different from theta
    theta_deg=math.degrees(theta)
    ang=theta_deg
    ydist=euclidianDistance(cp1,ap2)
    # Find qaudrant - Quadrant starts from the 2nd Geometrical quadrants and count increases clock wise The arrow must lie in that quadrant
    if (cp1[0]-ap2[0])>0 and (cp1[1]-ap2[1])>0:
        quad=1
    elif (cp1[0]-ap2[0])<0 and (cp1[1]-ap2[1])>0:
        quad=2
    elif (cp1[0]-ap2[0])<0 and (cp1[1]-ap2[1])<0:
        quad=3
    elif (cp1[0]-ap2[0])>0 and (cp1[1]-ap2[1])<0:
        quad=4
    # Compute angle to rotate
    print("theta degree",theta_deg,"quad",quad,"points Arrow",ap2,"points C",cp1)
    if(quad==1):
        x1 = math.cos(theta)
        x2 = math.sin(theta)
        x3 = -math.cos(theta)
        x4 = -math.sin(theta)
        x5 = -math.cos(theta)
        x6 =  -math.sin(theta)
        angleToRotateCW=270-ang
    elif(quad==3):
        theta =math.radians(ang-180) 
        x1 = -math.cos(theta)
        x2 = -math.sin(theta)
        x3 = math.cos(theta)
        x4 = math.sin(theta)
        x5 = math.cos(theta)
        x6 = math.sin(theta)
        angleToRotateCW=90-(ang-180)
    elif(quad==2):
        theta =math.radians(180-ang) 
        x1 = -math.cos(theta)
        x2 = math.sin(theta)
        x3 = math.cos(theta)
        x4 = -math.sin(theta)
        x5 = math.cos(theta)
        x6 = -math.sin(theta)
        angleToRotateCW=90+(180-ang)
    elif(quad==4):
        theta =math.radians(360-ang) 
        x1 = math.cos(theta)
        x2 = -math.sin(theta)
        x3 = -math.cos(theta)
        x4 = math.sin(theta)
        x5 = -math.cos(theta)
        x6 = math.sin(theta)
        angleToRotateCW=270+(360-ang)
    elif ang==0:
        x1 = 1
        x2 = 0
        x3 = -1
        x4 = 0
        x5 = -1
        x6 = 0
        angleToRotateCW=270
    elif ang==90:
        x1 = 0
        x2 = 1
        x3 = 0
        x4 = -1
        x5 = 0
        x6 = -1
        angleToRotateCW=180
    elif ang==180:
        x1 = 1
        x2 = 0
        x3 = -1
        x4 = 0
        x5 = -1
        x6 = 0
        angleToRotateCW=270
    elif ang==270:
        x1 = 0
        x2 = -1
        x3 = 0
        x4 = 1
        x5 = 0
        x6 = 1
        angleToRotateCW=0
    
    print(quad)
    print("Angle in degrees",angleToRotateCW)
    # angleradian = math.radians(angleToRotateCW)
    ac=euclidianDistance(ap2,cp1)
    ai=euclidianDistance(ap2,ip3)

    s1=ac_can_full/ac
    s2=ai_can_full/ai
    # scale=math.sqrt(s1*s2)
    # scale = (s1+s2)/2
    scale = s1
    A_offset_centre = 700/scale
    C_offset_centre = 556/scale
    I_offset_centre = 805/scale
    

    cent_proj_a = [ap2[0]+x1*A_offset_centre,ap2[1]+x2*A_offset_centre]
    cent_proj_c = [cp1[0]+x3*C_offset_centre,cp1[1]+x4*C_offset_centre]
    cent_proj_i = [ip3[0]+x5*I_offset_centre,ip3[1]+x6*I_offset_centre]

    cent = [(cent_proj_a[0]+cent_proj_c[0])/2,(cent_proj_a[1]+cent_proj_c[1])/2]
    print("Center A",cent_proj_a,"Center C",cent_proj_c,"Center I",cent_proj_i,"Center ",cent)
    centimg=[img.shape[1]/2,img.shape[0]/2]
    transxRight=int(centimg[0]-cent[0])
    transyDown=int(centimg[1]-cent[1])
    tranformedCenters=[0,0,0,0,0,0]
    for ind,cents in enumerate(centers):
#         print(ind)
        cents[0] = cents[0]+transxRight
        cents[1] = cents[1]+transyDown
        tranformedCenters[2*ind]=cents[0]
        tranformedCenters[2*ind+1]=cents[1]
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([ [1,0,transxRight], [0,1,transyDown] ])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    return angleToRotateCW,img_translation,scale,quad,tranformedCenters
#     print(ang,quad)

def returnROI(img,centers):
    """Return cropped RDT
        
        Args:

            img (numpy.ndarray) : Image with channels last format
            centers (list) : Centers of red and blue line (Used for debugging only)
        
        Returns:
            2-element tuple containing

                    - **image** (*numpy.ndarray*): RDT image 

                    - **centers** (*list*): List of transformed centers (Used for debugging only)

    """    
    startx=int(img.shape[0]-img.shape[0]/2-1000) 
    endx=int(img.shape[0] - img.shape[0]/2+1000)
    starty=int(img.shape[1]-img.shape[1]/2-50)
    endy=int(img.shape[1]-img.shape[1]/2+50)



    roi = img[startx:endx,starty:endy,:]
    tranformedCenters=[0,0,0,0,0,0]
    for ind,cents in enumerate(centers):
        cents[0]-=starty
        cents[1]-=startx
        tranformedCenters[2*ind]=cents[0]
        tranformedCenters[2*ind+1]=cents[1]

    if startx<0:
        startx=int(img.shape[1]-img.shape[1]/2-1000) 
        endx=int(img.shape[1] - img.shape[1]/2+1000)
        starty=int(img.shape[0]-img.shape[0]/2-50)
        endy=int(img.shape[0]-img.shape[0]/2+50)
        roi = img[starty:endy,startx:endx,:]
        for ind,cents in enumerate(centers):
            cents[0]-=starty
            cents[1]-=startx
            tranformedCenters[2*ind]=cents[0]
            tranformedCenters[2*ind+1]=cents[1]

        # cv2.rectangle(img,(startx,starty),(endx,endy), (0,0,255), 5)

    return roi,tranformedCenters


def rotate_bound(image, angle,centers):
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
    for ind,cents in enumerate(centers):

        tmpcent=[cents[0],cents[1]]
        cents[0]=M[0,0]*tmpcent[0] +M[0,1]*tmpcent[1] +M[0,2]
        cents[1]=M[1,0]*tmpcent[0] +M[1,1]*tmpcent[1]+M[1,2]
        tranformedCenters[2*ind]=cents[0]
        tranformedCenters[2*ind+1]=cents[1]
    return cv2.warpAffine(image, M, (nW, nH)),tranformedCenters


def postProcessDetections(labels):
    """PostProcess object detection output
        
        Args:

            labels (numpy.ndarray) : Bounding boxes of objects detected and the confidence score
            
        
        Returns:
       
            dict : Post processed detections
    """  
    
    result={}
    dictOfBoxesConf={}
    dictOfBoxesL={}
    for l in labels:
        try:
            if file_path_sets.YOLO_MODEL_VER==1:
                class_feat = str(int(l[-1]))
            elif file_path_sets.YOLO_MODEL_VER==2:
                class_feat = str(int(l[-1]/10))
            # print(class_feat)
            dictOfBoxesConf[class_feat].append(l[-2])
            dictOfBoxesL[class_feat].append([l[0],l[1],l[2],l[3]])
        except Exception as e:
            dictOfBoxesConf[class_feat]=[]
            dictOfBoxesL[class_feat]=[]
            dictOfBoxesConf[class_feat].append(l[-2])
            dictOfBoxesL[class_feat].append([l[0],l[1],l[2],l[3]])
    reducedL=reduceByConfidence(dictOfBoxesConf,dictOfBoxesL)
    try:
        centreTop = returnCentre(reducedL["0"])
        result["0"]=centreTop
    except Exception as e:
        # print(e)
        pass
    try:
        centreBottom = returnCentre(reducedL["2"])
        result["2"]=centreBottom
    except Exception as e:
        # print(e)
        pass
    try:
        centreTest = returnCentre(reducedL["1"])
        result["1"] = centreTest #[(int(reducedL["1"][0]),int(reducedL["1"][1])),(int(reducedL["1"][2]),int(reducedL["1"][3]))]
    except Exception as e:
        # print(e)
        pass

    return result

def angleOfLine(p1,p2):
    return math.atan2(p2[1]-p1[1],p2[0]-p1[0])

def angle_constraint(orientation,theta_deg):
    T=30
    d=abs(orientation-theta_deg)
    if(d>180): d=360-d
    if(d>T): return True
    return False

def warpPoint(point, R):
    result=[0,0]
    result[0] = point[0] * R[0,0] + point[1] *  R[0,1]+  R[0,2]
    result[1] = point[1] * R[1,0] + point[1] *  R[1,1]+  R[1,2]
    return result
    

def detect2(a, c, i):
    #rotation
    orientations=[a[-1],c[-1],i[-1]]
    a=np.array(a[:2])
    c=np.array(c[:2])
    i=np.array(i[:2])
    th1=angleOfLine(a,c)
    th2=angleOfLine(a,i)
    theta=(th1+th2)/2
    if(theta<0): theta+=2*math.pi
    #avoid feature orientations which are very different from theta
    theta_deg=math.degrees(theta)
    if(angle_constraint(orientations[0],theta_deg) or angle_constraint(orientations[1],theta_deg) or angle_constraint(orientations[2],theta_deg)):
        return MAX_VALUE
    
    ac=euclidianDistance(a,c)
    ai=euclidianDistance(a,i)

    s1=ac/ac_can
    s2=ai/ai_can
    scale=math.sqrt(s1*s2)

    #avoid scales which are very different from each other
    scale_disparity=s1/s2
    if(scale_disparity>1.25 or scale_disparity<0.75):
        return MAX_VALUE

    cos_th=math.cos(-1*theta)
    sin_th=math.sin(-1*theta)
    R= np.zeros((2,3))
    R[0,:]=np.array([cos_th/scale,-sin_th/scale,0])
    R[1,:]=np.array([sin_th/scale,cos_th/scale,0])

    #warp the points
    a1 = warpPoint(a,R)
    c1 = warpPoint(c,R)
    i1 = warpPoint(i,R)

    ac1_mid=[(a1[0]+c1[0])/2,(a1[1]+c1[1])/2]
    #translate back to 0,0
    a1=[a1[0]-ac1_mid[0],a1[1]-ac1_mid[1]]
    c1=[c1[0]-ac1_mid[0],c1[1]-ac1_mid[1]]
    i1=[i1[0]-ac1_mid[0],i1[1]-ac1_mid[1]]

    #compute the MSE
    return (euclidianDistance(ref_A,np.array(a1))+euclidianDistance(ref_C,np.array(c1))+euclidianDistance(ref_I,np.array(i1)))/3


def generateRDTcropV2(boxes,im0):

    """Generate RDT cropped image from object detection output
        
        Args:

            boxes (numpy.ndarray) : Bounding boxes of objects detected and the confidence score
            im0 (numpy.ndarray) : Input image
            targets (dict) : Centers of red and blue line (Used for debugging only)
        
        Returns:
       
            dict : Response with RDT crop if found
    """  
    min_error=MAX_VALUE
    BOX_A=[]
    BOX_C=[]
    BOX_I=[]
    A_best=[]
    C_best=[]
    I_best=[]
    orientationAngles=[0,22.5,45,135,157.5,180,202.5,225,315,337.5]
    for prediction in boxes:       
        class_feat =int(prediction[-1]/10)
        orientation = orientationAngles[int(prediction[-1]%10)]
        prediction=list(prediction)
        prediction.append(orientation)
        if class_feat==2:
            BOX_A.append(prediction)
        elif class_feat==1:
            BOX_C.append(prediction)
        elif class_feat==0:
            BOX_I.append(prediction)
    BOX_A=sorted(BOX_A, key = lambda x: x[4],reverse=True)
    BOX_C=sorted(BOX_C, key = lambda x: x[4],reverse=True)
    BOX_I=sorted(BOX_I, key = lambda x: x[4],reverse=True)
    if(len(BOX_A)>0 and len(BOX_C)>0 and len(BOX_I)>0):
    # print(BOX_A[0],BOX_C[0],BOX_I[0])
        for box_a in BOX_A:
            C_arrow_predicted=returnCentre(list(box_a[:4]))
            C_arrow_predicted.append(box_a[-1])
            for box_c in BOX_C:
                C_Cpattern_predicted=returnCentre(list(box_c[:4]))
                C_Cpattern_predicted.append(box_c[-1])
                for box_i in BOX_I:
                    C_Infl_predicted=returnCentre(list(box_i[:4]))
                    C_Infl_predicted.append(box_i[-1])

                    error=detect2(C_arrow_predicted,C_Cpattern_predicted,C_Infl_predicted)                
                    if error<min_error:
                        A_best=C_arrow_predicted
                        C_best=C_Cpattern_predicted
                        I_best=C_Infl_predicted
        
        angleToRotate,im0,scale_percent,quad,[cx_A,cy_A,cx_B,cy_B,cx_C,cy_C]=angle_with_yaxis2(np.array(C_best),np.array(A_best),np.array(I_best),im0,[[0,0],[0,0],[0,0]],0)
        cv2.imwrite("translated.jpg",im0)
        # Resize image
        print("scale",scale_percent)
        if scale_percent > 5:
                return [{"message":"Failure"},False]
        resizedImage = cv2.resize(im0, (int(im0.shape[1]*scale_percent),int(im0.shape[0]*scale_percent)))
        cv2.imwrite("resized.jpg",resizedImage)
        [cx_A,cy_A,cx_B,cy_B,cx_C,cy_C] = [cx_A*scale_percent,cy_A*scale_percent,cx_B*scale_percent,cy_B*scale_percent,cx_C*scale_percent,cy_C*scale_percent]

        # Rotate image

        rotatedImage,[cx_A,cy_A,cx_B,cy_B,cx_C,cy_C]=rotate_bound(resizedImage,angleToRotate,[[cx_A,cy_A],[cx_B,cy_B],[cx_C,cy_C]])
        cv2.imwrite("rotated.jpg",rotatedImage)
        # Pad image if the lowest dimension is less than 2000
        if rotatedImage.shape[0]<=2000:
            pad = (2000-rotatedImage.shape[0])/2
            tmp = np.zeros((2000,rotatedImage.shape[1],3))
            end=2000-pad
            tmp[int(pad):int(end),:,:]=rotatedImage
            rotatedImage=tmp
            [cx_A,cy_A,cx_B,cy_B,cx_C,cy_C] = [cx_A,cy_A+pad,cx_B,cy_B+pad,cx_C,cy_C+pad]

        if rotatedImage.shape[1]<=2000:
            pad = (2000-rotatedImage.shape[1])/2
            tmp = np.zeros((rotatedImage.shape[0],2000,3))
            end=2000-pad
            tmp[:,int(pad):int(end),:]=rotatedImage
            rotatedImage=tmp        
            [cx_A,cy_A,cx_B,cy_B,cx_C,cy_C] = [cx_A+pad,cy_A,cx_B+pad,cy_B,cx_C+pad,cy_C]
            
            # Generate RDT cropped image
        processed,[cx_A,cy_A,cx_B,cy_B,cx_C,cy_C]=returnROI(rotatedImage,[[cx_A,cy_A],[cx_B,cy_B],[cx_C,cy_C]])
        cv2.imwrite("rdt_crop.jpg",processed)
        return [{"message":"success"},processed]
    else:
        return [{"message":"Failure"},False]


def generateRDTcrop(boxes,im0,targets):
    """Generate RDT cropped image from object detection output
        
        Args:

            boxes (numpy.ndarray) : Bounding boxes of objects detected and the confidence score
            im0 (numpy.ndarray) : Input image
            targets (dict) : Centers of red and blue line (Used for debugging only)
        
        Returns:
       
            dict : Response with RDT crop if found
    """  
    
    res=postProcessDetections(boxes)
    featsPres = 0
    if (("2" in res.keys()) and ("0" in res.keys())) or (("2" in res.keys()) and ("1" in res.keys())) or (("1" in res.keys()) and ("0" in res.keys())):
        if (("2" in res.keys()) and ("0" in res.keys())):
            featsPres = 0
            x1y1 = np.array([int(res["0"][0]),int(res["0"][1])])
            x2y2 = np.array([int(res["2"][0]),int(res["2"][1])])
        elif (("1" in res.keys()) and ("0" in res.keys())):
            featsPres = 1
            x1y1 = np.array([int(res["0"][0]),int(res["0"][1])])
            x2y2 = np.array([int(res["1"][0]),int(res["1"][1])])
        elif (("2" in res.keys()) and ("1" in res.keys())):
            featsPres = 2
            x1y1 = np.array([int(res["1"][0]),int(res["1"][1])])
            x2y2 = np.array([int(res["2"][0]),int(res["2"][1])])
        try:
            cx,cy,w,h=[float(x) for x in targets["2"].split()]
            cx_C = im0.shape[1]*cx
            cy_C = im0.shape[0]*cy
        except :
            cx_C=0
            cy_C=0
        try:
            cx,cy,w,h=[float(x) for x in targets["0"].split()]
            cx_A = im0.shape[1]*cx
            cy_A = im0.shape[0]*cy
        except :
            cx_A=0
            cy_A=0
        try:
            cx,cy,w,h=[float(x) for x in targets["1"].split()]
            cx_B = im0.shape[1]*cx
            cy_B = im0.shape[0]*cy
        except :
            cx_B=0
            cy_B=0
        
        # Translate image and compute angle to rotate and scale factor.
        angleToRotate,im0,scale_percent,quad,[cx_A,cy_A,cx_B,cy_B,cx_C,cy_C]=angle_with_yaxis(x1y1,x2y2,im0,[[cx_A,cy_A],[cx_B,cy_B],[cx_C,cy_C]],featsPres)
        cv2.imwrite("translated.jpg",im0)
        # Resize image
        print("scale",scale_percent)
        if scale_percent > 5:
             return [{"message":"Failure"},False]
        resizedImage = cv2.resize(im0, (int(im0.shape[1]*scale_percent),int(im0.shape[0]*scale_percent)))
        cv2.imwrite("resized.jpg",resizedImage)
        [cx_A,cy_A,cx_B,cy_B,cx_C,cy_C] = [cx_A*scale_percent,cy_A*scale_percent,cx_B*scale_percent,cy_B*scale_percent,cx_C*scale_percent,cy_C*scale_percent]

        # Rotate image

        rotatedImage,[cx_A,cy_A,cx_B,cy_B,cx_C,cy_C]=rotate_bound(resizedImage,angleToRotate,[[cx_A,cy_A],[cx_B,cy_B],[cx_C,cy_C]])
        cv2.imwrite("rotated.jpg",rotatedImage)
        # Pad image if the lowest dimension is less than 2000
        if rotatedImage.shape[0]<=2000:
            pad = (2000-rotatedImage.shape[0])/2
            tmp = np.zeros((2000,rotatedImage.shape[1],3))
            end=2000-pad
            tmp[int(pad):int(end),:,:]=rotatedImage
            rotatedImage=tmp
            [cx_A,cy_A,cx_B,cy_B,cx_C,cy_C] = [cx_A,cy_A+pad,cx_B,cy_B+pad,cx_C,cy_C+pad]

        if rotatedImage.shape[1]<=2000:
            pad = (2000-rotatedImage.shape[1])/2
            tmp = np.zeros((rotatedImage.shape[0],2000,3))
            end=2000-pad
            tmp[:,int(pad):int(end),:]=rotatedImage
            rotatedImage=tmp        
            [cx_A,cy_A,cx_B,cy_B,cx_C,cy_C] = [cx_A+pad,cy_A,cx_B+pad,cy_B,cx_C+pad,cy_C]
            
        # Generate RDT cropped image
        processed,[cx_A,cy_A,cx_B,cy_B,cx_C,cy_C]=returnROI(rotatedImage,[[cx_A,cy_A],[cx_B,cy_B],[cx_C,cy_C]])
        cv2.imwrite("rdt_crop.jpg",processed)

        return [{"message":"success"},processed]
    else:
        return [{"message":"Failure"},False]


class FluServer:
    def __init__(self):
        '''FluServer init. handler for the two models'''
        print("Creating flu server")
        self.__yolo = YOLO()
        self.__lineDetector = LineDetector()

    def callyolo(self, image):
        print("calling yolo")
        return self.__yolo.wrapper(image)

    def callLineDetector(self, image):

        return self.__lineDetector.wrapper(image)



    def ret_inf_lat(self):
        '''Returns the last TFS(EI) inf time delta.'''
        return str(self.__yolo.grpc_delta+self.__lineDetector)



def runPipeline(img,serverObj):
    boxes = serverObj.callyolo(img)
    im0 = np.copy(img)
    im0 = utils.draw_bbox(im0, boxes, show_label=True)
    resp,roi = generateRDTcrop(boxes,img,[])

    rc = -4
    if resp["message"]=="success":
        cv2.imwrite("roi.jpg", roi[1000:1500,:,:])
        outImage,virus_type,blue_detection = serverObj.callLineDetector(roi)
        print(virus_type,blue_detection)
        try:
            if blue_detection>0 and virus_type==0:
                rc=0
            elif blue_detection ==0:
                message="No_control_line"
                rc=-1
            elif virus_type==1:
                rc = 1
                message="Atype"
            elif virus_type==2:
                message="Btype"
                rc = 2
            elif virus_type==3:
                message="A+Btype"
                rc = 3
                
        except IndexError:
            pass
        cv2.imwrite("out.jpg",outImage[1000:1500,:,:])
    else:
        message="No rdt found"
        rc = -2
    return rc

def processRdtRequest(UUID,include_proof,img_str,serv):
    '''
        Reads rdt input image and tells the result: No rdt, No flu, Type A flu, Type B flu. Handles errors as well.
        This function is called from the rest API code which extracts the required data from the request and calls this function.
    '''
    print("Calling fn to process rdt in flasker")
    message1="Negative"
    rc=0
    try:
        nparr = np.fromstring(img_str, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
        org_h, org_w, _ = img_np.shape
        if file_path_sets.YOLO_MODEL_VER==1:
            pass
        elif file_path_sets.YOLO_MODEL_VER==2:
            print("height",org_h,"weight",org_w)
            if (org_h>org_w):
                img_np=cv2.transpose(img_np)
                img_np=cv2.flip(img_np,flipCode=0)

        print("Reading rdt img")
    except IOError:
        print("Unable to open rdt jpeg")
    
    im0 = np.copy(img_np)
    st = time.time()
    boxes = serv.callyolo(img_np)
    et = time.time()
    t1=et-st
    _img_ = np.copy(im0)
    # colCorr=(0.011307082205892827, -0.014425887586575259, 0.3093816717263326)
    # img_ = img_/255.0
    # img_ = np.array(img_,dtype=np.float32)
    # imgYUV=cv2.cvtColor(img_,cv2.COLOR_BGR2YUV)

    
    # imgYUV[:,:,0]=imgYUV[:,:,0]-(colCorr[2]/2)
    # imgYUV[:,:,1]=imgYUV[:,:,1]-(colCorr[0]*2)
    # imgYUV[:,:,2]=imgYUV[:,:,2]-(colCorr[1]*2)
    # imgGainLow = cv2.cvtColor(imgYUV,cv2.COLOR_YUV2BGR)

    # cv2.imwrite( "boostedV.jpg" ,imgGainLow*255)
    # _img_ =cv2.imread("boostedV.jpg")    
    try:
        im0 = utils.draw_bbox(im0, boxes, show_label=True)
        print("Util processing img")
    except IOError:
        print("Image reading error")
    if file_path_sets.YOLO_MODEL_VER==1:
        resp,roi = generateRDTcrop(boxes,_img_,[])
    else:
        resp,roi =generateRDTcropV2(boxes,_img_)

    if resp["message"]=="success": 
            try:
                # postprocessed=enhanceImage(roi[1000:1500,:,:])
                postprocessed=gaussBlur(roi[1000:1500,:,:])
                # postprocessed= enhanceImage(postprocessed)
                cv2.imwrite("roi_gausian.jpg", postprocessed)
                postprocessed=postprocessed*1.2
                cv2.imwrite("roi_gausian_amp1pnt2.jpg", postprocessed)
                
                postprocessed= enhanceImage(roi[1000:1500,:,:])
                postprocessed=gaussBlur(postprocessed)
                cv2.imwrite("roi_enhan_gausian.jpg", postprocessed)
                # postprocessed=gaussBlur(roi[1000:1500,:,:])
                # postprocessed= enhanceImage(postprocessed)
                cv2.imwrite("roi.jpg", roi[1000:1500,:,:])
            

                print("Overwrite roi jpeg")

            except IOError:
                print("Unable to open roi jpeg")
            st=time.time()         
            outImage,virus_type,blue_detection = serv.callLineDetector(roi)     
            et=time.time()
            t2=t1+et-st
            try:
                if blue_detection>0 and virus_type==0:
                    rc=0
                elif blue_detection ==0:
                    message1="No_control_line"
                    rc=-1
                elif virus_type==1:
                    rc = 1
                    message1="Atype"
                elif virus_type==2:
                    message1="Btype"
                    rc = 2
                elif virus_type==3:
                    message1="A+Btype"
                    rc = 3
            except IndexError:
                pass
            cv2.imwrite("out.jpg",outImage[1000:1500,:,:])
            print("Time taken",t2)
    else:
            print("No rdt found")
            message1="No rdt found"
            rc = -2
    
    if include_proof=="True" and rc != -2:
            try:
                with open("roi.jpg", "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                    m = MultipartEncoder(fields={'metadata': ('response',json.dumps({"UUID":UUID,"rc":str(rc),"msg":message1,"Include Proof":include_proof}),'application/json'),
                    'image': ('rdt', encoded_string, 'image/jpeg')})
                    return m,True,rc
            except IOError:
                print("Unable to open roi image")
                raise IOError('Unable to open roi file')
                
    elif include_proof=="True" and rc==-2:
            m = MultipartEncoder(fields={'metadata': ('response',json.dumps({"UUID":UUID,"rc":str(rc),"msg":message1,"Include Proof":include_proof}),'application/json')})
            return m,True,rc
    else:
            encoded_string=None
            resp = json.dumps({"UUID":UUID,"rc":str(rc),"msg":message1,"Include Proof":include_proof})
            return resp,False,rc
            



