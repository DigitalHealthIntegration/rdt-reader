from django.shortcuts import render
from django.http import HttpResponse
from django import forms
from api.settings import RDT_GIT_ROOT
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
        print("Form content: {0}".format(form))
        print("Process form content")
        if form.is_valid():
            try:
                md=form['metadata'].value()
                print(md)
                UUID=json.loads(md)["UUID"]
                print(UUID)
                include_proof=json.loads(md)["Include_Proof"]
                print(include_proof)
                message="No Flu"
                rc = 0
                imagefile=request.FILES['image']
                img_str=imagefile.read()
                imagefile.close()
                m,retFlag,rc = flasker.processRdtRequest(UUID,include_proof,img_str)
                if retFlag==True:
                    print(m.to_string)
                    boundary = m.boundary
                    print("Boundary value: "+boundary)
                    r1 = m.fields.get("metadata")
                    finalRspTxt = "\n"+boundary+"\nContent-Disposition: form-data; name=\"metadata\"; filename=\""+r1[0]+"\"\nContent-Type: "+r1[2]+"\r\n\r\n"+r1[1]
                    print(finalRspTxt)
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
                    print("IOError raised while processing rdt - ")
                    return HttpResponse("<h1>Rdt Post IO error</h1>",status="400")
            except ValueError:
                    return HttpResponse("<h1>Rdt internal Post failure</h1>",status="400")
                            
        else:
            return HttpResponse("<h1>Rdt Post failure</h1>",status="400")
    else:
        return HttpResponse("<h1>Request not supported</h1>",status="405")

@csrf_exempt
def DoHealthCheck(request):
    '''API endpoint to verify the service is up and running.
       Input alias: /health-check (GET)
    '''
    if request.method == 'GET':
        return HttpResponse("OK",content_type="application/json")
    else:
        return HttpResponse("<h1>Request not supported</h1>",status="405")

  
    
