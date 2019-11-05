import cv2
import sys
import imgaug.augmenters as iaa

sys.path.insert(1,"D:\\source\\repos\\rdt-reader\\object_detection_v2")
import core.model_new as model
from core.config import cfg
import numpy as np
from utils import data_loader
inpImg="../object_detection_mobile_v2/train_hor_ratioCropped/I4.jpg"
import ntpath
import math
import itertools 
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import tensorflow as tf
def euclidianDistance(p1,p2):
    """Compute euclidian distance between p1 and p2
        
        Args:

            p1 (numpy.array) : X,Y of point 1
            p2 (numpy.array) : X,Y of point 2
        
            
        Returns:
       
            numpy.float: Distance between two points
    """
    return np.linalg.norm(p2-p1)

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
    # img = cv2.pyrDown(img)
    img = img[...,np.newaxis]
    img = img/255.0
    img = img[np.newaxis,...]
    img = np.array(img,dtype=np.float32)
    return img


def main():
    num_class = cfg.TRAIN.NUMBER_CLASSES
    resize_dim=cfg.TEST.INPUT_SIZE
    anchors=cfg.TRAIN.ANCHOR_ASPECTRATIO
    number_blocks = cfg.TRAIN.NUMBER_BLOCKS
    resizefactor = [0,0]
    resizefactor[0] = int(resize_dim[0]/number_blocks[0])
    resizefactor[1] = int(resize_dim[1]/number_blocks[1])
    Model = model.ObjectDetection(True,"Model_KH_EXP/model_save_rot_360x640.hdf5").model
    
    with open(cfg.TEST.LABEL_FILE_YOLO) as fin, open("Analysis_2.csv","w") as fout:
        # fout.write("Arrow`_prob,Arrow`_cx,Arrow`_cy,Arrow_cx,Arrow_cy,Arrow`_Angle,Arrow_Angle,Cpattern`_prob,Cpattern`_cx,Cpattern`_cy,Cpattern_cx,Cpattern_cy,Cpattern`_Angle,Cpattern_Angle,Inlfuenza`_prob,Inlfuenza`_cx,Inlfuenza`_cy,Inlfuenza_cx,Inlfuenza_cy,Inlfuenza`_Angle,Inlfuenza_Angle\n")
        fout.write("ImageName,Arrow`_prob,Cpattern`_prob,Inlfuenza`_prob,A_ang-A`_ang,C_ang-C`_ang,I_ang-I`_ang-,A_C,C_I,A_I,A-A`,C-C`,I-I`\n")

        print(cfg.TEST.LABEL_FILE_YOLO)
        for line in fin:
            imgpath=line.strip().split()[0]
            print(imgpath)
            trueArrow=[0,0]
            trueCpattern=[0,0]
            trueInfl=[0,0]
            for annots in line.strip().split()[1:]:
                x1y1x2y2=[float(x) for x in annots.split(",")[:-1]]
                lab = int(annots.split(",")[-1])
                feat_type = int(lab/10)

                cxywh=data_loader.xy2cxcy(x1y1x2y2)
                if feat_type==0:
                    trueInfl[0]=cxywh[0]
                    trueInfl[1]=cxywh[1]
                elif feat_type==1:
                    trueCpattern[0]=cxywh[0]
                    trueCpattern[1]=cxywh[1]
                
                elif feat_type==2:
                    trueArrow[0]=cxywh[0]
                    trueArrow[1]=cxywh[1]

            imgName = ntpath.basename(imgpath)
            orientation=int(imgName.split("_")[0])
            
            fullsizeimg = cv2.imread(imgpath,cv2.IMREAD_COLOR)
            KPS=[
                Keypoint(x=trueArrow[0],y=trueArrow[1]),
                Keypoint(x=trueCpattern[0],y=trueCpattern[1]),
                Keypoint(x=trueInfl[0],y=trueInfl[1])
            ]
            kpsoi = KeypointsOnImage(KPS, shape=fullsizeimg.shape)

            fullsizeimg,kps_aug = iaa.Affine(rotate=(-10,10))(image=fullsizeimg,keypoints=kpsoi)
            
            trueArrow,trueCpattern,trueInfl=[kps_aug.keypoints[0].x,kps_aug.keypoints[0].y],[kps_aug.keypoints[1].x,kps_aug.keypoints[1].y],[kps_aug.keypoints[2].x,kps_aug.keypoints[2].y]
            final_img = np.zeros((fullsizeimg.shape[0]*4+80,fullsizeimg.shape[1],3))        
            # print(img)
            final_img[0:fullsizeimg.shape[0],:fullsizeimg.shape[1],:]=fullsizeimg
            final_img[fullsizeimg.shape[0]+20:fullsizeimg.shape[0]*2+20,:fullsizeimg.shape[1],:]=fullsizeimg*0.2
            final_img[fullsizeimg.shape[0]*2+20:fullsizeimg.shape[0]*3+20,:fullsizeimg.shape[1],:]=fullsizeimg*0.2
            final_img[fullsizeimg.shape[0]*3+20:fullsizeimg.shape[0]*4+20,:fullsizeimg.shape[1],:]=fullsizeimg*0.2

            Input = prepocessImageCOD(fullsizeimg,resize_dim)
            # print(np.max(Input))
            interpreter = tf.lite.Interpreter(model_path="D:/source/repos/object_detection_mobile_v2/eval_model/OD.lite")
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_shape = input_details[0]['shape']
            print(input_details)
            interpreter.set_tensor(input_details[0]['index'], Input)
            interpreter.invoke()
            # 

            predictions = interpreter.get_tensor(output_details[0]['index'])
            # predictions=Model.predict(Input) 190 10 x 19
            preds = predictions #np.reshape(predictions,(predictions.shape[0],number_blocks[0],number_blocks[1],4,num_class+4))
            preds=preds[0]
            all_boxes = []
            orie=[0,1,2,6,7,8,9,10,14,15]
            
            orie_angles=[0,22.5,45,135,157.5,180,202.5,225,315,337.5]

            orientation=orie.index(orientation)
            orientation_angle=orie_angles[orientation]
            for ax_1 in range(number_blocks[0]): #10
                for ax_2 in range(number_blocks[1]): #19
                    for anch_id in range(len(anchors[0])):
                        computedIndex=ax_1*number_blocks[1]+ax_2
                        tar_class = np.argmax(preds[computedIndex,anch_id,0:num_class])
                        # print(preds[ax_1,ax_2,anch_id,0:num_class])
                        
                        prob=preds[computedIndex,anch_id,tar_class]
                        offsets = preds[computedIndex,anch_id,num_class:]
                        cx = (ax_2+0.5)*resizefactor[1]+offsets[-4]*resize_dim[1]
                        cy =  (ax_1+0.5)*resizefactor[0]+offsets[-3]*resize_dim[0]
                        w = anchors[0][anch_id][1]*math.exp(offsets[-2])
                        h = anchors[0][anch_id][0]*math.exp(offsets[-1])
                        # if(tar_class==20):
                        #     print(cx,cy,w,h,prob,ax_1,ax_2,anch_id,offsets)
                        x1,y1,x2,y2=data_loader.cxcy2xy([cx,cy,w,h])
                        all_boxes.append([tar_class,x1,y1,x2,y2,prob])
            
            widthFactor = 1.0/resize_dim[1]*fullsizeimg.shape[1]
            heightFactor = 1.0/resize_dim[0]*fullsizeimg.shape[0]
            fullsizeimg=fullsizeimg*0.1
            all_boxes=sorted(all_boxes,key=lambda x: x[5],reverse=True) 
            Boxes_Arrow=[]
            Boxes_Cpattern=[]
            Boxes_Infl=[]

            for b in all_boxes:
                tar=int(b[0])
                color_ind=int(tar/10)
                predicted_orientatation=tar%10
                predicted_orientatation=orie_angles[predicted_orientatation]
                
                x1=int(b[1]*widthFactor)
                y1_true=int(b[2]*heightFactor)
                y1=int(b[2]*heightFactor)+(fullsizeimg.shape[0]*(color_ind+1)+20)
                x2=int(b[3]*widthFactor)
                y2=int(b[4]*heightFactor)+(fullsizeimg.shape[0]*(color_ind+1)+20)
                y2_true=int(b[4]*heightFactor)
                val=int(127*b[5]+128)
                cxcy=data_loader.xy2cxcy([x1,y1_true,x2,y2_true])
                if predicted_orientatation==orientation_angle:
                    if color_ind!=3:
                        for i in range(3):
                            if i ==color_ind:
                                final_img[y1:y2,x1:x2,color_ind]=val
                            else:
                                final_img[y1:y2,x1:x2,i]=0
                if(b[5]>0.0):
                    if color_ind==2:
                        Boxes_Arrow.append([b[5],cxcy[0],cxcy[1],trueArrow[0],trueArrow[1],predicted_orientatation,orientation_angle])
                    elif color_ind==1:
                        Boxes_Cpattern.append([b[5],cxcy[0],cxcy[1],trueCpattern[0],trueCpattern[1],predicted_orientatation,orientation_angle])
                    elif color_ind==0:
                        Boxes_Infl.append([b[5],cxcy[0],cxcy[1],trueInfl[0],trueInfl[1],predicted_orientatation,orientation_angle])
            all_box_preds=[Boxes_Arrow,Boxes_Cpattern,Boxes_Infl]
            all_combinations=list(itertools.product(*all_box_preds))
            # sorted_all_combinations=sorted(all_combinations, key=lambda x: x[0][0],reverse=True)
            # print(sorted_all_combinations[0])
            print(len(Boxes_Arrow),len(Boxes_Cpattern),len(Boxes_Infl))
            print(Boxes_Arrow)
            for cmbs in all_combinations:
                fout.write(imgName+",")
                cmb_f=[]
                for cmb in cmbs:
                    
                    cmb_f += [x for x in cmb]
                    fout.write(str(cmb[0])+",")
                # print(cmb_f)
                Apred_A=euclidianDistance(np.array(cmb_f[1],cmb_f[2]),np.array(cmb_f[3],cmb_f[4]))
                Cpred_C=euclidianDistance(np.array(cmb_f[8],cmb_f[9]),np.array(cmb_f[10],cmb_f[11]))
                Ipred_I=euclidianDistance(np.array(cmb_f[15],cmb_f[16]),np.array(cmb_f[17],cmb_f[18]))
                A_C = euclidianDistance(np.array(cmb_f[1],cmb_f[2]),np.array(cmb_f[8],cmb_f[9]))
                C_I = euclidianDistance(np.array(cmb_f[15],cmb_f[16]),np.array(cmb_f[8],cmb_f[9]))
                A_I = euclidianDistance(np.array(cmb_f[1],cmb_f[2]),np.array(cmb_f[15],cmb_f[16]))
                Apredang_Aang=cmb_f[6]-cmb_f[5]
                Cpredang_Cang=cmb_f[13]-cmb_f[12]
                Ipredang_Iang=cmb_f[20]-cmb_f[19]
                cmb_str=",".join([str(v) for v in [Apredang_Aang,Cpredang_Cang,Ipredang_Iang,A_C,C_I,A_I,Apred_A,Cpred_C,Ipred_I]])
                fout.write(cmb_str)
                fout.write("\n")
                
            cv2.imwrite("heatmap/"+imgName,final_img)
            break
    # print(box_0,box_1)
main()





