# -*- coding: utf-8 -*-
"""Flask server
Example:

        $ python flasker.py
"""
from flask import Flask, request
from tf_model_api import YOLO,LineDetector
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
from flask import Flask, Response
from requests_toolbelt import MultipartEncoder
import base64
import time
import os.path
from os import path




app = Flask(__name__)

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
        if l[-1]!=3:
            try:
                dictOfBoxesConf[str(int(l[-1]))].append(l[-2])
                dictOfBoxesL[str(int(l[-1]))].append([l[0],l[1],l[2],l[3]])
            except Exception as e:
                dictOfBoxesConf[str(int(l[-1]))]=[]
                dictOfBoxesL[str(int(l[-1]))]=[]
                dictOfBoxesConf[str(int(l[-1]))].append(l[-2])
                dictOfBoxesL[str(int(l[-1]))].append([l[0],l[1],l[2],l[3]])
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
        self.__yolo = YOLO()
        self.__lineDetector = LineDetector()

    def callyolo(self, image):

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
                message="No control line found"
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


@app.route("/health-check", methods=["GET"])
def health_check():
    '''API endpoint to verify the service is up and running.'''
    if request.method == "GET":
        return Response("OK", mimetype="application/json")


@app.route("/Quidel/QuickVue", methods=["POST"])
def interpret_quidel_quickvue():
    '''API endpoint to Run the entire service and give appropriate response.

        Example:
                
            Sample Request::

                    {"UUID":"a43f9681-a7ff-43f8-a1a6-f777e9362654","Quality_parameters":{"brightness":"10"},"RDT_Type":"Flu_Audere","Include_Proof":"False"}

        Response codes
            
            0=> No Flu detected
            
            1=> Type A Flu detected
            
            2=> Type B Flu detected
            
            3=> Both type A and B detected
            
            Negative values indicate error conditions
            
            -1=> Invalid(No Control Line detected)
            
            -2=> No RDT found in image

    
        Example:

            Sample API response:: 
                    
                    {"UUID":"a43f9681-a7ff-43f8-a1a6-f777e9362654",rc":0,"msg":"No Flu","Include_Proof":"False"}
    '''
    t1=0
    t2=0
    UUID=json.loads(request.form.get("metadata"))["UUID"]
    include_proof = json.loads(request.form.get("metadata"))["Include_Proof"]
    
    if request.method == "POST":
        message="No Flu"
        rc = 0 
        imagefile=request.files.get('image') # Get image from file
        img_str = imagefile.read()
        imagefile.close()
        nparr = np.fromstring(img_str, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
        im0 = np.copy(img_np)
        st = time.time()
        boxes = serv.callyolo(img_np)
        et = time.time()
        t1=et-st
        im0 = utils.draw_bbox(im0, boxes, show_label=True)
        resp,roi = generateRDTcrop(boxes,img_np,[])
        if resp["message"]=="success": 
            cv2.imwrite("roi.jpg", roi[1000:1500,:,:])
            st=time.time()
            outImage,virus_type,blue_detection = serv.callLineDetector(roi)
            et=time.time()
            t2=t1+et-st
            try:
                if blue_detection>0 and virus_type==0:
                    rc=0
                elif blue_detection ==0:
                    message="No control line found"
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
        print("Time taken",t2)
        if include_proof=="True" and path.exists('roi.jpg') :
            with open("roi.jpg", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
        
            m = MultipartEncoder(
                   fields={'metadata': ('response',json.dumps({"UUID":UUID,"rc":str(rc),"msg":message,"Include Proof":include_proof}),'application/json'),
                'image': ('rdt', encoded_string, 'image/jpeg')})
            return Response(m.to_string(), mimetype=m.content_type)

        else:
            resp = json.dumps({"UUID":UUID,"rc":str(rc),"msg":message,"Include Proof":include_proof})
            return Response(resp, mimetype="application/json")


@app.route("/align", methods=["POST"])
def align():
    '''Backwards compatible alias for interpret_quidel_quickvue().

       This can be deleted once all code migrates to the new path.
    '''
    interpret_quidel_quickvue()


if __name__ == "__main__":
    serv = FluServer()
    app.run(host='0.0.0.0', port=9000)
