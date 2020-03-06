import os
from multiprocessing import Pool,Lock
import cv2
import json
import ntpath
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from flasker import angle_with_yaxis,returnCentre
import numpy as np
from sklearn.cluster import KMeans

folderDirectory="../object_detection_mobile_v2/dataset/images_seg_te/"
onlyfiles = [os.path.join(folderDirectory, f) for f in os.listdir(folderDirectory) if os.path.isfile(os.path.join(folderDirectory, f))]
rootPathCentreLabel="../object_detection_mobile_v2/dataset/labels_seg_tr/"
classes=["top","bottom","2","7","1"]
outdirectory ="../object_detection_mobile_v2/output_check/"
horizontalImages="../object_detection_mobile_v2/train_16_rotations_ratioCropped/"
weirdRotations=["IMG_1614.jpg"]
def returnAugmentationObjRot(angle):
    seq = iaa.Sequential(
        [
            
            iaa.Affine(
                rotate=(angle), # rotate by -x to +x degrees
            ),

        ])
    return seq
def returnAugmentationObj(angle,Scale):
    seq = iaa.Sequential(
        [
            
            iaa.Affine(
                rotate=(angle), # rotate by -x to +x degrees
                scale={"x": (Scale), "y": (Scale)}
            ),

        ])
    return seq
def returnAugmentationObjCrop(top,right,bott,left):
    seq = iaa.Sequential(
        [
            
            iaa.CropAndPad(px=(-top,
                -right,
                -bott,
                -left),
                keep_size=False,
            ),
            # iaa.Resize((256,256))
        ])
    return seq

def processImage(imagePath,all_annots_imagelist):
    imageFileName=ntpath.basename(imagePath)

    all_annots,imagelist=all_annots_imagelist[0],all_annots_imagelist[1]
    
    resize_dim=(256,256)
    img = cv2.imread(imagePath,cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_size=img.shape
    # img = cv2.resize(img,resize_dim)
    try:
        objects = all_annots[imagelist[imageFileName]]["frames"][imageFileName]
        rotateAndAug(img,objects,imageFileName)
    except KeyError:
        print(imageFileName,"No frames found")
    # lock.release()

def rotateAndAug(img,objs,imageFileName):
    newImg=img
    newObjects=objs
    BBoxs=[BoundingBox(x1=0, x2=0, y1=0, y2=0),BoundingBox(x1=0, x2=0, y1=0, y2=0),BoundingBox(x1=0, x2=0, y1=0, y2=0),BoundingBox(x1=0, x2=0, y1=0, y2=0)]
    p1=[0,0]
    p2=[0,0]
    for inde,obj in enumerate(objs):
        if obj["tags"][0] in classes:
            x1 = obj["box"]["x1"]
            y1 = obj["box"]["y1"]
            x2 = obj["box"]["x2"]
            y2 = obj["box"]["y2"]

            if obj["tags"][0] in ["top","2"]:
                BBoxs[0]=BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2,label="top")
                p1 = returnCentre([x1,y1,x2,y2])
            elif obj["tags"][0] in ["bottom","7"]:
                BBoxs[1]=BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2,label="bottom")
                p2 = returnCentre([x1,y1,x2,y2])
            elif obj["tags"][0] in ["1"]:
                BBoxs[2]=BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2,label="influenza")
                p3 = returnCentre([x1,y1,x2,y2])
            BBoxs[3]=BoundingBox(x1=0, x2=obj["width"], y1=0, y2=obj["height"],label="entireimage")
    if max(p1)>0 and max(p2)>0:
        p1=np.array(p1)
        p2=np.array(p2)
        angleToRotate,im0,scale_percent,quad,_=angle_with_yaxis(p1,p2,img,[],0)
        angleToRotate=angleToRotate+90
        seqAug = returnAugmentationObj(angleToRotate,scale_percent)
        bbs = BoundingBoxesOnImage(BBoxs, shape=img.shape)
        newshape = [img.shape[0]*scale_percent,img.shape[1]*scale_percent]
        image_hor, bbs_hor = seqAug(image=img, bounding_boxes=bbs)

        X=max(bbs_hor.bounding_boxes[-1].x1,bbs_hor.bounding_boxes[-1].x2)
        x1_inf=bbs_hor.bounding_boxes[0].x1+118
        y1_inf=bbs_hor.bounding_boxes[0].y1
        x2_inf=min(bbs_hor.bounding_boxes[0].x2+150,X)
        y2_inf=bbs_hor.bounding_boxes[0].y2
        listOfbb=bbs_hor.bounding_boxes
        imggrayaug=cv2.cvtColor(image_hor, cv2.COLOR_BGR2GRAY)
        infCalculatedFlag=False

        if (y2_inf-y1_inf)>70:
            for i in range(3):
                yy=listOfbb[i].y2
                y=listOfbb[i].y1
                offset=((yy-y)-70)/2
                
                listOfbb[i].y2-=offset
                listOfbb[i].y1+=offset
            offset=((y2_inf-y1_inf)-70)/2
            y2_inf-=offset
            y1_inf+=offset
        
        if listOfbb[2].label==None:
            # print((x2_inf-x1_inf),imageFileName)
            if (x2_inf-x1_inf)>102:
                listOfbb[2]=(BoundingBox(x1=x1_inf,x2=x2_inf,y1=y1_inf,y2=y2_inf))
                bbs_hor =BoundingBoxesOnImage(listOfbb, shape=image_hor.shape)
                infCalculatedFlag=True
                listOfbb[2].label="influenza"
            elif (x2_inf-x1_inf)>0 and (x2_inf-x1_inf)<102:
                # print(removeInf,image_hor.shape,int(x2_inf-x1_inf))

                image_hor[:,int(x1_inf):int(x2_inf),:]=0
                # print(image_hor[:,removeInf:,:])
                # crop_right=int(x2_inf-x1_inf)
                # seqAug2 = returnAugmentationObjCrop(0,crop_right,0,0)
                # image_hor, bbs_hor =seqAug2(image=image_hor, bounding_boxes=bbs_hor)

        
        cnt=0
        for numRotation in range(16):
            if numRotation in [0,1,2,8,7,6,10,9,15,14]:
                
                angleToRotate=numRotation*22.5
                seqAug = returnAugmentationObjRot(angleToRotate)
                image_aug_, bbs_aug_ = seqAug(image=image_hor, bounding_boxes=bbs_hor)
                topBB=bbs_aug_.bounding_boxes[0]
                bottBB=bbs_aug_.bounding_boxes[1]
                p1=returnCentre([topBB.x1,topBB.y1,topBB.x2,topBB.y2])
                p2=returnCentre([bottBB.x1,bottBB.y1,bottBB.x2,bottBB.y2])
                centreRDT=returnCentre([p2[0],p2[1],p1[0],p1[1]])
                # print(centreRDT)
                leftLim=centreRDT[0]-640
                rightLim=centreRDT[0]+640
                topLim=centreRDT[1]-360
                botLim=centreRDT[1]+360
                crop_top=int(topLim) #int(Imagebb.y1)
                crop_right=int(-rightLim+image_aug_.shape[1])#int(-Imagebb.x2+image_aug.shape[1])
                crop_bott=int(-botLim+image_aug_.shape[0])#int(-Imagebb.y2+image_aug.shape[0])
                crop_left=int(leftLim)#int(Imagebb.x1)
                seqAug2 = returnAugmentationObjCrop(crop_top,crop_right,crop_bott,crop_left)
                image_aug, bbs_aug =seqAug2(image=image_aug_, bounding_boxes=bbs_aug_)

                image_with_bbs = bbs_aug.draw_on_image(image_aug)
                
    ###########UNCOMMENT FOR CREATING YOLO TRAINING FILES
                lock.acquire()
                with open("rdt_test_crop_rot.txt","a") as fout:
                    boxes=bbs_aug.bounding_boxes
                    fout.write(os.path.join(horizontalImages,str(numRotation)+"_"+imageFileName)+" ")
                    for bb in boxes:
                        if bb.label =="influenza":
                            annots=[str(x) for x in [bb.x1,bb.y1,bb.x2,bb.y2]]
                            annots=",".join(annots)+","+str(cnt)+" "
                            fout.write(annots)
                        if bb.label =="top":
                            annots=[str(x) for x in [bb.x1,bb.y1,bb.x2,bb.y2]]
                            annots=",".join(annots)+","+str(10+cnt)+" "
                            fout.write(annots)
                        if bb.label =="bottom":
                            annots=[str(x) for x in [bb.x1,bb.y1,bb.x2,bb.y2]]
                            annots=",".join(annots)+","+str(10*2+cnt)+" "
                            fout.write(annots)
                    fout.write("\n")
                cv2.imwrite(os.path.join(horizontalImages,str(numRotation)+"_"+imageFileName),image_aug)
                cv2.imwrite(os.path.join(outdirectory,str(numRotation)+"_"+"rot"+imageFileName),image_with_bbs)
                cnt+=1
                with open("anchors.txt","a") as fout:
                    for bb in bbs_aug.bounding_boxes[:-1]:
                        
                        w=bb.x2-bb.x1
                        h=bb.y2-bb.y1
                        # bbox_size.append(np.array([w,h]))
                        fout.write(str(w)+","+str(h)+"\n")
                lock.release()
###################END BLOCK

    return newImg,newObjects

def rotate(img,objs,imageFileName):
    newImg=img
    newObjects=objs
    BBoxs=[BoundingBox(x1=0, x2=0, y1=0, y2=0),BoundingBox(x1=0, x2=0, y1=0, y2=0),BoundingBox(x1=0, x2=0, y1=0, y2=0),BoundingBox(x1=0, x2=0, y1=0, y2=0)]
    p1=[0,0]
    p2=[0,0]
    for inde,obj in enumerate(objs):
        if obj["tags"][0] in classes:
            x1 = obj["box"]["x1"]
            y1 = obj["box"]["y1"]
            x2 = obj["box"]["x2"]
            y2 = obj["box"]["y2"]

            if obj["tags"][0] in ["top","2"]:
                BBoxs[0]=BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2,label="top")
                p1 = returnCentre([x1,y1,x2,y2])
            elif obj["tags"][0] in ["bottom","7"]:
                BBoxs[1]=BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2,label="bottom")
                p2 = returnCentre([x1,y1,x2,y2])
            elif obj["tags"][0] in ["1"]:
                BBoxs[2]=BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2,label="influenza")
                p3 = returnCentre([x1,y1,x2,y2])
            BBoxs[3]=BoundingBox(x1=0, x2=obj["width"], y1=0, y2=obj["height"],label="entireimage")
    if max(p1)>0 and max(p2)>0:
        p1=np.array(p1)
        p2=np.array(p2)
        angleToRotate,im0,scale_percent,quad,_=angle_with_yaxis(p1,p2,img,[],0)
        angleToRotate=angleToRotate+90
        seqAug = returnAugmentationObj(angleToRotate,scale_percent)
        bbs = BoundingBoxesOnImage(BBoxs, shape=img.shape)
        newshape = [img.shape[0]*scale_percent,img.shape[1]*scale_percent]
        image_aug, bbs_aug = seqAug(image=img, bounding_boxes=bbs)
        Imagebb=bbs_aug.bounding_boxes[-1]
        topBB=bbs_aug.bounding_boxes[0]
        bottBB=bbs_aug.bounding_boxes[1]
        p1=returnCentre([topBB.x1,topBB.y1,topBB.x2,topBB.y2])
        p2=returnCentre([bottBB.x1,bottBB.y1,bottBB.x2,bottBB.y2])
        centreRDT=returnCentre([p2[0],p2[1],p1[0],p1[1]])
        # print(centreRDT)
        leftLim=centreRDT[0]-640
        rightLim=centreRDT[0]+640
        topLim=centreRDT[1]-360
        botLim=centreRDT[1]+360
        crop_top=int(topLim) #int(Imagebb.y1)
        crop_right=int(-rightLim+image_aug.shape[1])#int(-Imagebb.x2+image_aug.shape[1])
        crop_bott=int(-botLim+image_aug.shape[0])#int(-Imagebb.y2+image_aug.shape[0])
        crop_left=int(leftLim)#int(Imagebb.x1)
        X=max(bbs_aug.bounding_boxes[-1].x1,bbs_aug.bounding_boxes[-1].x2)
        x1_inf=bbs_aug.bounding_boxes[0].x1+152
        y1_inf=bbs_aug.bounding_boxes[0].y1
        x2_inf=min(bbs_aug.bounding_boxes[0].x2+170,X)
        y2_inf=bbs_aug.bounding_boxes[0].y2
        listOfbb=bbs_aug.bounding_boxes
        imggrayaug=cv2.cvtColor(image_aug, cv2.COLOR_BGR2GRAY)
        infCalculatedFlag=False

        if (y2_inf-y1_inf)>90:
            for i in range(3):
                yy=listOfbb[i].y2
                y=listOfbb[i].y1
                offset=((yy-y)-90)/2
                
                listOfbb[i].y2-=offset
                listOfbb[i].y1+=offset
            offset=((y2_inf-y1_inf)-90)/2
            y2_inf-=offset
            y1_inf+=offset
        
        if listOfbb[2].label==None:

            listOfbb[2]=(BoundingBox(x1=x1_inf,x2=x2_inf,y1=y1_inf,y2=y2_inf))
            bbs_aug =BoundingBoxesOnImage(listOfbb, shape=image_aug.shape)
            infCalculatedFlag=True
            listOfbb[2].label="influenza"

        histg = cv2.calcHist([imggrayaug[int(y1_inf):int(y2_inf),int(x1_inf):int(bbs_aug.bounding_boxes[0].x2+170)]],[0],None,[2],[0,256]) 
        # print(X)
        factor=2
        if histg[1]!=0:
            factor=histg[0]/histg[1]
        else:
            pass

        if infCalculatedFlag:
            if factor<2:
                # listOfbb[2]=BoundingBox(x1=x1_inf,y1=y1_inf,x2=x2_inf,y2=y2_inf)
                bbs_aug =BoundingBoxesOnImage(listOfbb, shape=image_aug.shape)
                seqAug2 = returnAugmentationObjCrop(crop_top,crop_right,crop_bott,crop_left)
                image_aug, bbs_aug =seqAug2(image=image_aug, bounding_boxes=bbs_aug)
                image_with_bbs = bbs_aug.draw_on_image(image_aug)
                cv2.imwrite(os.path.join(outdirectory,"rot"+imageFileName),image_with_bbs)
            else:
                listOfbb[2]=BoundingBox(x1=0,y1=0,x2=0,y2=0,label=None)
                bbs_aug =BoundingBoxesOnImage(listOfbb, shape=image_aug.shape)
                seqAug2 = returnAugmentationObjCrop(crop_top,crop_right,crop_bott,crop_left)
                image_aug, bbs_aug =seqAug2(image=image_aug, bounding_boxes=bbs_aug)
                image_with_bbs = bbs_aug.draw_on_image(image_aug)
                cv2.imwrite(os.path.join(outdirectory,"rot"+imageFileName),image_with_bbs)
        else:
            # bbs_aug =BoundingBoxesOnImage(listOfbb, shape=image_aug.shape)
            seqAug2 = returnAugmentationObjCrop(crop_top,crop_right,crop_bott,crop_left)
            image_aug, bbs_aug =seqAug2(image=image_aug, bounding_boxes=bbs_aug)            
            image_with_bbs = bbs_aug.draw_on_image(image_aug)
            cv2.imwrite(os.path.join(outdirectory,"rot"+imageFileName),image_with_bbs)
        # image_with_bbs = bbs.draw_on_image(img)
###########UNCOMMENT FOR CREATING YOLO TRAINING FILES
        lock.acquire()
        with open("rdt_train_crop.txt","a") as fout:
            boxes=bbs_aug.bounding_boxes
            fout.write(os.path.join(horizontalImages,imageFileName)+" ")
            for bb in boxes:
                if bb.label =="influenza":
                    annots=[str(x) for x in [bb.x1,bb.y1,bb.x2,bb.y2]]
                    annots=",".join(annots)+",0 "
                    fout.write(annots)
                if bb.label =="top":
                    annots=[str(x) for x in [bb.x1,bb.y1,bb.x2,bb.y2]]
                    annots=",".join(annots)+",1 "
                    fout.write(annots)
                if bb.label =="bottom":
                    annots=[str(x) for x in [bb.x1,bb.y1,bb.x2,bb.y2]]
                    annots=",".join(annots)+",2 "
                    fout.write(annots)
            fout.write("\n")
        cv2.imwrite(os.path.join(horizontalImages,imageFileName),image_aug)
        
        with open("anchors.txt","a") as fout:
            for bb in bbs_aug.bounding_boxes[:-1]:
                
                w=bb.x2-bb.x1
                h=bb.y2-bb.y1
                # bbox_size.append(np.array([w,h]))
                fout.write(str(w)+","+str(h)+"\n")
        lock.release()
###################END BLOCK

    return newImg,newObjects

def init(l):
    global lock
    lock = l

def main():
    l = Lock()
    all_annotations = []
    image_list_id = {}
    for ind,element in enumerate(os.listdir(rootPathCentreLabel)):
        with open(os.path.join(rootPathCentreLabel,element)) as fin:
            annotations = json.load(fin)
            all_annotations.append(annotations)
            for f in annotations["frames"].keys():
                image_list_id[f]=ind
    args = [(x,[all_annotations,image_list_id]) for x in onlyfiles]
    p = Pool(8,initializer=init, initargs=(l,))
    p.starmap(processImage,args)
    p.close()
    p.join()

if __name__ == "__main__":
    anchors=[]
    main()


    with open("anchors.txt") as fin:
        for line in fin:
            line=line.strip().split(",")
            anchors.append([float(line[0]),float(line[1])])

    # print(bbox_size)
    anchors=np.array(anchors)
    kmeans = KMeans(n_clusters=9)
    kmeans.fit(anchors)
    centroids = kmeans.cluster_centers_
    print(centroids)
    print("Done")