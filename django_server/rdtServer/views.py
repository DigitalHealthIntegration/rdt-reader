from django.shortcuts import render
from django.http import HttpResponse
from django import forms
from api.settings import RDT_GIT_ROOT,SERV,BUCKET
from .forms import RequestForm
from rest_framework import generics
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Align
from .serializers import AlignSerializer
from django.views.decorators.csrf import csrf_exempt
import json
import sys
import logging
import numpy as np
import cv2
sys.path.append(RDT_GIT_ROOT)
import flasker


@csrf_exempt
def ViewRdt(request):
    '''API endpoint to Run the entire service and give appropriate response.
        Input alias: /Quidel/QuickVue/ (POST)
                     /alias/  (POST) - may be deprecated.
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
    if request.method == 'POST':
        form = RequestForm(request.POST,request.FILES)
        #form = RequestForm(request.POST)
        logging.debug("Form content: {0}".format(form))
        if form.is_valid():
            try:
                md=form['metadata'].value()
                logging.debug(md)
                UUID=json.loads(md)["UUID"]
                logging.debug(UUID)
                include_proof=json.loads(md)["Include_Proof"]
                message="No Flu"
                rc = 0
                imagefile=request.FILES['image']
                img_str=imagefile.read()
                imagefile.close()
                nparr = np.fromstring(img_str, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
                
                logging.info(img_str)
                m,retFlag,rc = flasker.processRdtRequest(UUID,include_proof,img_str,SERV)
                if retFlag==True:
                    # print(m.to_string)
                    boundary = m.boundary
                    r1 = m.fields.get("metadata")
                    finalRspTxt = "\n"+boundary+"\nContent-Disposition: form-data; name=\"metadata\"; filename=\""+r1[0]+"\"\nContent-Type: "+r1[2]+"\r\n\r\n"+r1[1]
                    logging.debug(finalRspTxt)
                    if rc != -2:
                        #this means rdt found
                        r2 = m.fields.get("image")
                        finalRspImg = "\n"+boundary+"\nContent-Disposition: form-data; name=\"image\"; filename=\""+r2[0]+"\"\nContent-Type: "+r2[2]+"\r\n\r\n"+r2[1].decode("utf-8")
                        fullRsp = finalRspTxt+"\n"+finalRspImg+"\n"+boundary
                    else:
                        fullRsp = finalRspTxt+"\n"+boundary
                    
                    finalRsp = HttpResponse(fullRsp,content_type="multipart/form-data; boundary="+boundary,status="200")
                    return finalRsp
                    #return HttpResponse(finalRsp,status="200")
                else:
                    return HttpResponse(m, content_type="application/json")
            except IOError as ioe:
                    logging.error("IOError raised while processing rdt")
                    return HttpResponse("<h1>Rdt Post IO error</h1>",status="400")
            except ValueError:
                    logging.error("ValError raised while processing rdt")
                    return HttpResponse("<h1>Rdt internal Post failure</h1>",status="400")
                            
        else:
            logging.error("Invalid http POST form data")
            return HttpResponse("<h1>Rdt Post failure</h1>",status="400")
    else:
        logging.error("Unsupported http request")
        return HttpResponse("<h1>Request not supported</h1>",status="405")

@csrf_exempt
def HIVRdt(request):
    '''API endpoint to Run the entire service and give appropriate response.
        Input alias: /WITS/HIV/ (POST)
                     /alias/  (POST) - may be deprecated.
        Example:
                
            Sample Request::
                    {"UUID":"a43f9681-a7ff-43f8-a1a6-f777e9362654","RDT_Type":"HIV"}
        Response codes
            
            0=> unsuccesful upload to s3
            
            1=> succesful upload s3
            
        Example:
            Sample API response:: 
                    
                    {"UUID":"a43f9681-a7ff-43f8-a1a6-f777e9362654",rc":0}
    '''
    resp={}
    if request.method == 'POST':
        form = RequestForm(request.POST,request.FILES)
        #form = RequestForm(request.POST)
        logging.debug("Form content: {0}".format(form))
        if form.is_valid():
            try:
                md=form['metadata'].value()
                logging.debug(md)
                UUID=json.loads(md)["UUID"]
                logging.debug(UUID)
                include_proof=json.loads(md)["Include_Proof"]
                rc = 0
                imagefile=request.FILES['image']
                img_str=imagefile.read()
                imagefile.close()
                nparr = np.fromstring(img_str, np.uint8)
                labelfile=request.FILES['label']
                label_str=labelfile.read()
                labelfile.close()
                print(label_str)

                label_json= json.loads(label_str)
                print(label_json)
                nparr = np.fromstring(img_str, np.uint8)
                with open("label.json","w") as fout:
                    fout.write(json.dumps(label_json))
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
                cv2.imwrite("image.jpg",img_np)
                resp["UUID"]=UUID
                res1 = flasker.upload_file("image.jpg", BUCKET,"WITS_HIV_RDT/images/"+UUID+".jpg")
                res2 = flasker.upload_file("label.json", BUCKET,"WITS_HIV_RDT/labels/"+UUID+".json")
                
                if(res1 and res2):
                    resp["rc"]=1
                else:
                    resp["rc"]=0
                
                
                return HttpResponse(resp, content_type="application/json")
            except IOError as ioe:
                    logging.error("IOError raised while processing rdt")
                    return HttpResponse("<h1>Rdt Post IO error</h1>",status="400")
            except ValueError:
                    logging.error("ValError raised while processing rdt")
                    return HttpResponse("<h1>Rdt internal Post failure</h1>",status="400")
                            
        else:
            logging.error("Invalid http POST form data")
            return HttpResponse("<h1>Rdt Post failure</h1>",status="400")
    else:
        logging.error("Unsupported http request")
        return HttpResponse("<h1>Request not supported</h1>",status="405")



@csrf_exempt
def DoHealthCheck(request):
    '''API endpoint to verify the service is up and running.
       Input alias: /health-check (GET)
    '''
    if request.method == 'GET':
        return HttpResponse(json.dumps({"status":"OK"}),content_type="application/json")
    else:
        return HttpResponse("<h1>Request not supported</h1>",status="405")

  
    
