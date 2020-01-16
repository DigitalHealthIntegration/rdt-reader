# -*- coding: utf-8 -*-
"""Train a CNN to detect presence of red and blue line on an RDT and also give the normalized y-axis location.

Example:

        $ python train_blue_red.py --transfer_learning True

"""
import os
import numpy as np
import cv2
import imgaug as ia
import keras.backend as K
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage
from sklearn.utils import class_weight
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda, AvgPool2D
from keras.models import Sequential,Model
from keras.layers import ReLU, Dense, Conv2D, Flatten,Dropout, MaxPooling2D, GlobalAveragePooling3D, LeakyReLU, Activation, BatchNormalization, Input, merge, Softmax
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import random
from keras.constraints import max_norm
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3,preprocess_input
import argparse
from mixup_generator import MixupGenerator
from tensorflow.keras import layers
import keras
def normalize(data):
    return (data - data.mean()) / data.std()
def loadData(noBatchSamples,batchIdx,duplicativeFactor=1,rareData="faint.txt",badData="baddata.txt",rootPathCentreLabel="./obj/labels",rootPathCroppedImages = "./obj/images"):
    """This function loads data from the directory of labels, it works with the yolo data format.
        
        Args:

            noBatchSamples (int) : Number of samples per batch
            batchIdx (int) : Batch number
            duplicativeFactor (int) : Number of times to oversample rare date
            rareData (str) : Filenames of rare data samples
            rootPathCentreLabel (str) : Directory with labels in yolo format
            rootPathCroppedImages (str) : Directory with images, image name and label name should be same eg: 1.jpg 1.txt
        
        Returns:
            list: Images in list 
            list: Targets 
            list: File names

    """
    
    y_train=[]
    X_train=[]
    x_faint = []
    y_faint = []
    name=[]
    name_faint=[]
    test_data=[]
    f = open(rareData)
    lines = f.readlines()
    lines = [x.strip() for x in lines]
    f.close()
    f = open(badData)
    lines_bad = f.readlines()
    lines_bad = [x.strip() for x in lines_bad]
    f.close()
    blue=0
    faintInd=0
    for ind,element in enumerate(os.listdir(rootPathCentreLabel)):
        if  ind >= noBatchSamples*batchIdx and ind<=noBatchSamples*(batchIdx+1) and element.replace(".txt","") not in lines_bad:
            with open(os.path.join(rootPathCentreLabel,element)) as fin:
                y = [(0,0),(0,0)]
                img = cv2.imread(os.path.join(rootPathCroppedImages,element.replace(".txt",".jpg")),cv2.IMREAD_COLOR)
                #img  = gaussBlur(img)
                #img = enhanceImage(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img[:,20:80,:]
                for line in fin:
                    line=line.split(" ")
                    if line[0]=="0" :
                        blue+=1
                        pixel_y_pos =int(float(line[2]))
                        y[0]=(int(float(line[1])*100),int(float(line[2])*2000))
                    elif line[0]=="1" :
                        pixel_y_pos =int(float(line[2]))
                        if float(line[2])<1:
                            y[1]=(int(float(line[1])*100),int(float(line[2])*2000))
                    elif line[0]=="2" :
                        pixel_y_pos =int(float(line[2]))
                        if float(line[2])<1:
                            y[1]=(int(float(line[1])*100),int(float(line[2])*2000))
            if element.replace(".txt","") in lines:
                x_faint.append(img)
                y_faint.append(y)
                name_faint.append(element)
            else:
                y_train.append(y)
                X_train.append(img)
                name.append(element)
        else:
            test_data.append(element)
    with open("test_data.txt","w") as fout:
        fout.write("\n".join(test_data))
#     X_train=np.array(X_train,dtype=np.uint8)
#     x_faint =np.array(x_faint,dtype=np.uint8)
    print("Number of blue",blue)
    return X_train,y_train,name,name_faint,x_faint,y_faint #np.zeros((128, 32, 32, 3), dtype=np.uint8) + (batch_idx % 255)

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

def returnLOGker():
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

def LOG(im,kernel):
    img = im/255.0
    img = np.array(img,dtype=np.float32)
    imgYUV=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    imgYUV[:,:,1:]=imgYUV[:,:,1:]-0.5
    filtered_img_GB=cv2.filter2D(imgYUV , cv2.CV_32F, kernel)*255
    return filtered_img_GB[:,:,1:]




def gaborFilt(im):
    g_kernel = cv2.getGaborKernel((9, 51), 6, np.pi/2, 0.2, 0.1, np.pi, ktype=cv2.CV_32F)
    img = im/255.0
    img = np.array(img,dtype=np.float32)
    imgYUV=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

    filtered_img_GB = cv2.filter2D(imgYUV*255, cv2.CV_32F, g_kernel)
    renormalized=np.zeros((imgYUV.shape))
    # renormalized=renormalized[:,5:95,:]

    renormalized[:,:,0]=renormalize(filtered_img_GB[:,:,0],(np.min(filtered_img_GB[:,:,0]),np.max(filtered_img_GB[:,:,0])),(0,255))
    renormalized[:,:,1]=renormalize(filtered_img_GB[:,:,1],(np.min(filtered_img_GB[:,:,1]),np.max(filtered_img_GB[:,:,1])),(-128,127))
    renormalized[:,:,2]=renormalize(filtered_img_GB[:,:,2],(np.min(filtered_img_GB[:,:,2]),np.max(filtered_img_GB[:,:,2])),(-128,127))
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

    newIMG[:,:,0]=renormalize(renormalized[:,:,0],(np.min(renormalized[:,:,0]),np.max(renormalized[:,:,0])),(0,255))
    newIMG[:,:,1]=renormalize(Final_U,(np.min(Final_U),np.max(Final_U)),(0,255))
    newIMG[:,:,2]=renormalize(Final_V,(np.min(Final_V),np.max(Final_V)),(0,255))
    im = newIMG[:,:,1:]
    return im

def gaussBlur(img):
    img = cv2.GaussianBlur(img,(1,11),0)
    return img

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def enhanceImage(img):
    img = np.uint8(img)
    newimg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    clahe = cv2.createCLAHE(10, (5,5))
    newimg[1]=cv2.normalize(newimg[1], 0, 255, cv2.NORM_MINMAX)
    lab_planes = cv2.split(newimg)
    lab_planes[1] = clahe.apply(lab_planes[1])
    lab = cv2.merge(lab_planes)
    result=cv2.cvtColor(lab, cv2.COLOR_HLS2RGB)
    result = adjust_gamma(result,0.7)
    return result


def key2Target(keypoints,name):
    """This function converts keypoints returned after data augmentation to numpy arrays.
        
        Args:

            keypoints (imgaug.augmentables.kps.KeypointsOnImage) : Keypoints on the image
            name (list) : File names
        
        Returns:
            list: Images in list 
            list: Targets 
            list: File names

    """
    numred=0
    numblue=0
    y_test_regression=[]
    y_test_categorical = []
    for i,k in enumerate(keypoints):
        y=np.zeros((2))
        y_class=np.zeros((2))
        
        if k[0][1]<=700 and (k[1][1]<=700 or k[1][1]>=1800): # Red line:False && Blue line:False
            y[0]=0
            y[1]=0
            y_class[0]=0
            y_class[1]=0
            print(name[i],k)
        elif k[0][1]>=700 and (k[1][1]<=700 or k[1][1]>=1800): # Red line:False && Blue line:True
            numblue+=1
            y[0]=(k[0][1]-1000)/500
            y[1]=0
            y_class[0]=1
            y_class[1]=0
        elif (k[1][1]>=700 and k[1][1]<=1900) and k[0][1]>=700: # Red line:True && Blue line:True
            numblue+=1
            numred+=1
            y[0]=(k[0][1]-1000)/500
            y[1]=(k[1][1]-1000)/500
            y_class[0]=1
            y_class[1]=1
        else:
            print(name[i])
        y_test_regression.append(np.array(y))
        y_test_categorical.append(np.array(y_class))
    
    print("number of blue",numblue,"number of red",numred)
    return np.array(y_test_regression),np.array(y_test_categorical)


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
#             sometimes(iaa.Affine(
#                 translate_percent={"x": (-0.06, 0.06)}, # translate by -x to +x percent (per axis)
#                 rotate=(-5, 5) # rotate by -x to +x degrees
#             )),
            # execute 0 to 2 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
#             iaa.Dropout(0.02, name="Dropout"),
            iaa.Add((-5,5),per_channel=0.5),
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.AddToHueAndSaturation((-5, 5),per_channel=0.5), # change hue and saturation

            iaa.SomeOf((0, 2),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 4.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 8)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 13)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.GammaContrast((0.8,1.2),per_channel=True),
                    #iaa.OneOf([
                    #    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.016))), # Add perscpective transform
                    #    iaa.Affine(rotate=(-2, 2),scale=(0.75,1.2)) # rotate by -x to +x degrees


                    #]),
                                    ],
                random_order=True
            )
        ])
    return seq


def lossReg(y_true,y_pred):
    """Custom loss function to penalize A type virus versus B type written for keras.
    """
    mask=K.ones_like(y_true)
    l=K.square(y_pred-y_true)
    penalty = tf.constant([10.0])
    mask =tf.add(penalty,tf.to_float (tf.math.logical_or(tf.math.logical_and(tf.math.greater(y_true[:,0],y_true[:,1]),tf.math.less(y_pred[:,0],y_pred[:,1])),tf.math.logical_and(tf.math.less(y_true[:,0],y_true[:,1]),tf.math.greater(y_pred[:,0],y_pred[:,1])))))
    mask = tf.stack([K.ones_like(y_true[:,0]),mask],axis=1)
    return K.mean(tf.math.multiply(l,mask),axis=-1)
       
def returnModel(loadWeights,weightsFile="./red_blue.hdf5"):
    """This function returns a keras model.
        
        Args:
            loadWeights (bool) : Load weights specified in the weightsFile param
            weightsFile (str) : Path to weights
        
        Returns:
            :class:`keras.model.Model` : Neural Network 

    """
    x = Input(shape=(500, 100,3))

    conv1=Conv2D(8, (3,3), padding='valid')(x)
    batchnorm1 = BatchNormalization()(conv1)
    act1 = ReLU()(batchnorm1)
    

    conv2=Conv2D(8, (3,3), padding='valid')(act1)
    batchnorm2 = BatchNormalization()(conv2)
    act2 = ReLU()(batchnorm2)
    maxpool2 = MaxPooling2D((2,2))(act2)

    conv3=Conv2D(16, (3,3), padding='valid')(maxpool2)
    batchnorm3 = BatchNormalization()(conv3)
    act3 = ReLU()(batchnorm3)

    conv4=Conv2D(16, (3,3), padding='valid')(act3)
    batchnorm4 = BatchNormalization()(conv4)
    act4 = ReLU()(batchnorm4)
    maxpool3 = MaxPooling2D((2,2))(act4)

    flat1 = Flatten()(maxpool3)
    D1 = Dense(256)(flat1)
    batchnorm5 = BatchNormalization()(D1)
    act5 = ReLU()(batchnorm5)

    D2 = Dense(128,kernel_constraint=max_norm(2))(act5)
    batchnorm6 = BatchNormalization()(D2)
    act6 = ReLU()(batchnorm6)


    D_soft = Dense(2)(act6)
    batchnorm7 = BatchNormalization()(D_soft)
    out1 = Activation('sigmoid',name="cat_kash")(batchnorm7)

    D_sigmoid_blue = Dense(1)(act6)
    batchnorm8 = BatchNormalization()(D_sigmoid_blue)
    out_blue = Activation('sigmoid',name="reg_blue")(batchnorm8)
    
    D_sigmoid_red = Dense(1)(act6)
    batchnorm9 = BatchNormalization()(D_sigmoid_red)
    out_red = Activation('sigmoid',name="reg_red")(batchnorm9)
    
    model = Model(inputs=x, outputs=[out1,out_blue,out_red])
    if (loadWeights):
        model.load_weights(weightsFile,by_name=True)


    return model

def res_net_block(input_data, filters, conv_size):
     x = layers.Conv2D(filters, conv_size, activation='relu', padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None),kernel_regularizer=keras.regularizers.l2(l=0.00001))(input_data)
     x = layers.BatchNormalization()(x)
     x = layers.Conv2D(filters, conv_size, activation=None, padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None),kernel_regularizer=keras.regularizers.l2(l=0.00001))(x)
     x = layers.BatchNormalization()(x)
     x = layers.Add()([x, input_data])
     x = layers.Activation('relu')(x)
     return x


def modelShredding(loadWeights,weightsFile="./red_blue_shred.hdf5"):
    """This function returns a keras model.
        
        Args:
            loadWeights (bool) : Load weights specified in the weightsFile param
            weightsFile (str) : Path to weights
        
        Returns:
            :class:`keras.model.Model` : Neural Network 

    """
    x = Input(shape=(20,60,2))
    conv1=Conv2D(64, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None),kernel_regularizer=keras.regularizers.l2(l=0.0001))(x)
    conv1=Conv2D(64, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None),kernel_regularizer=keras.regularizers.l2(l=0.0001))(conv1)
    batchnorm1 = BatchNormalization()(conv1)
    act1 = ReLU()(batchnorm1)

    conv2=Conv2D(128, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None),kernel_regularizer=keras.regularizers.l2(l=0.001))(act1)
    conv2=Conv2D(128, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None),kernel_regularizer=keras.regularizers.l2(l=0.001))(conv2)
    batchnorm2 = BatchNormalization()(conv2)
    act2 = ReLU()(batchnorm2)
    #num_res_net_blocks = 4
    #for i in range(num_res_net_blocks):
    #    act2 = res_net_block(act2, 32, 3)

    conv3=Conv2D(256, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None),kernel_regularizer=keras.regularizers.l2(l=0.001))(act2)
    conv3=Conv2D(256, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None),kernel_regularizer=keras.regularizers.l2(l=0.001))(conv3)
    batchnorm3 = BatchNormalization()(conv3)
    act3 = ReLU()(batchnorm3)

    conv4=Conv2D(512, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None),kernel_regularizer=keras.regularizers.l2(l=0.001))(act3)
    conv4=Conv2D(512, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None),kernel_regularizer=keras.regularizers.l2(l=0.001))(conv4)
    batchnorm4 = BatchNormalization()(conv4)
    act4 = ReLU()(batchnorm4)
    #num_res_net_blocks = 4
    #for i in range(num_res_net_blocks):
    #    act4 = res_net_block(act4, 64, 3)
    globalAveragePooling=GlobalAveragePooling2D()(act4)
    D1 = Dense(256)(globalAveragePooling)
    batchnorm3 = BatchNormalization()(D1)
    act3 = ReLU()(batchnorm3)


    predictor = Dense(3)(act3)
    batchnorm9 = BatchNormalization()(predictor)
    output = Activation('softmax',name="classification")(batchnorm9)
    model = Model(inputs=x, outputs=output)
    if (loadWeights):
        model.load_weights(weightsFile,by_name=True)


    return model


def modelTransferLearning(loadWeights,weightsFile="./red_blue_transf.hdf5"):
    """This function returns a keras model with a pretrained Inception v3 being fine tuned and used as the feature extractor.
        
        Args:
            loadWeights (bool) : Load weights specified in the weightsFile param
            weightsFile (str) : Path to weights
        
        Returns:
            :class:`keras.model.Model` : Neural Network 

    """
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    D1 = Dense(256)(x)
    batchnorm5 = BatchNormalization()(D1)
    act5 = ReLU()(batchnorm5)
    D2 = Dense(128)(act5)
    batchnorm6 = BatchNormalization()(D2)
    act6 = ReLU()(batchnorm6) 
    D_soft = Dense(2)(act6)
    batchnorm7 = BatchNormalization()(D_soft)
    out1 = Activation('sigmoid',name="cat_kash")(batchnorm7)

    D_sigmoid = Dense(2)(act6)
    batchnorm8 = BatchNormalization()(D_sigmoid)
    out2 = Activation('sigmoid',name="reg_kash")(batchnorm8)

    model = Model(inputs=base_model.input, outputs=[out1,out2])
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    if (loadWeights):
        model.load_weights(weightsFile,by_name=True)
    return model



def convertProblem2Shred(X,Y,Aug,LOGker):
    x_shred=[]
    y_cat = []
    redFound=0
    y_pos_red=0
    y_pos_blue=0
    si=0
    ei=0
    numberofBlue=0
    numberofRed=0
    numberofNone=0
    Ynp=np.array(Y)
    for ind in range(len(X)):
#         img = cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)
#         img = img*255
        img=X[ind]
        for index in range(25):
            startIndex=index*20
            endIndex=(index+1)*20
            #print(startIndex)
            if Y[ind][1]*500>startIndex and Y[ind][1]*500<=endIndex and Y[ind][1]>0:
                numberofRed+=1
                y_pos_red=int(Y[ind][1]*500)
                redFound = 1
                si=y_pos_red-10
                ei=y_pos_red+10
#                 cv2.imwrite("./sample/train/"+str(ind)+"_"+str(index)+"red"+".jpg",img[si:ei,:,:])
                if True:
                    for shift in range(11):
                        tmp_aug_st = si + random.randint(-5,5)
                        tmp_aug_en = tmp_aug_st+20
                        if shift != 0:
                            img_aug=Aug(image=img)
                        else:
                            img_aug=img
                        img_aug = normalize(LOG(img_aug,LOGker))/128.0
                        x_shred.append(img_aug[tmp_aug_st:tmp_aug_en,:,:])
                        y_cat.append(np.array([0,1,0]))    
#                     cv2.imwrite("./sample/train/"+str(ind)+"_"+str(index)+"_"+str(shift)+"red"+".jpg",img[tmp_aug_st:tmp_aug_en,:,:])
            elif Y[ind][0]*500>startIndex and Y[ind][0]*500<=endIndex and Y[ind][0]>0:
                numberofBlue+=1
                y_pos_blue=int(Y[ind][0]*500)
                si=y_pos_blue-10
                ei=y_pos_blue+10
                if True:
                    for shift in range(10):
                        tmp_aug_st = si+random.randint(-5,5) 
                        tmp_aug_en = tmp_aug_st+20
                        if shift != 0:
                            img_aug=Aug(image=img)
                        else:
                            img_aug=img
                        img_aug = normalize(LOG(img_aug,LOGker))/128.0
                        x_shred.append(img_aug[tmp_aug_st:tmp_aug_en,:,:])
                        y_cat.append(np.array([1,0,0]))
    #                     cv2.imwrite("./sample/train/"+str(ind)+"_"+str(index)+"_"+str(shift)+"blue"+".jpg",img[tmp_aug_st:tmp_aug_en,:,:])

                    
#                 if numberofRed<=numberofBlue:
#                     numberofBlue+=1
# #                     cv2.imwrite("./sample/train/"+str(ind)+"_"+str(index)+"blue"+".jpg",img[si:ei,:,:])
            else:
#                 if numberofBlue+20>=numberofNone:
#                     numberofNone+=1
#                     cv2.imwrite("./sample/train/"+str(ind)+"_"+str(index)+"none"+".jpg",img[startIndex+10:endIndex-10,:,:])
                numberofNone+=1
                for shift in range(1):
                    tmp_aug_st = startIndex+random.randint(-1,1) 
                    tmp_aug_en = tmp_aug_st+20
                    if shift != 0:
                        img_aug=Aug(image=img)
                    else:
                        img_aug=img
                    img_aug = normalize(LOG(img_aug,LOGker))/128.0
                    if img_aug[tmp_aug_st:tmp_aug_en,:,:].shape[0]==20:
                        x_shred.append(img_aug[tmp_aug_st:tmp_aug_en,:,:])
                        y_cat.append(np.array([0,0,1]))
                    
        if True:
#             img=cv2.circle(img,(50,y_pos_red),5,(0,0,255),5)
#             img=cv2.circle(img,(50,si),2,(0,255,0),2)
#             img=cv2.circle(img,(50,ei),2,(0,255,0),2)
#             img=cv2.circle(img,(50,y_pos_blue),5,(255,0,0),5)
            
#             cv2.imwrite("./sample/train/"+str(ind)+"_"+str(index)+"full"+".jpg",img)
            redFound=0
    x_shred=np.array(x_shred)
    y_cat = np.array(y_cat)
    
    combined = list(zip(x_shred,y_cat))
    random.shuffle(combined)
    print("number of red found",numberofRed,"Number of blue found",numberofBlue,"Number of none",numberofNone,"Number of images",len(X))
    x_shred[:], y_cat[:] = zip(*combined)
    return x_shred,y_cat

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def arguementsHandler():
    parser = argparse.ArgumentParser(description='Train a CNN to detect presence of red and blue line on an RDT and also give the normalized y-axis location.')
    parser.add_argument('--transfer', default="false",
                        help='Set boolean to true to train a transfer learning model',type=str)
    parser.add_argument('--shred', default="true",
                        help='Set boolean to true to convert the problem into classifying shreds',type=str)

    try:
        args=parser.parse_args()
    except SystemExit:
        args=parser.parse_args(["--transfer true --shred true"])
    return args

if __name__ == "__main__":
    args = arguementsHandler()
    font = cv2.FONT_HERSHEY_SIMPLEX
    if args.transfer =="true" or args.transfer =="True":
        useTransferLearning = True
    else:
        useTransferLearning = False
    if args.shred =="true" or args.shred =="True":
        useShred = True
    else:
        useShred = False

    if useTransferLearning:
        print("\n\nCurrently using transfer learning...\n\n")
    else:
        print("\n\nCurrently using custom model...\n\n")
    LOGker=returnLOGker()
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) # Correctly put number of GPU and CPU
    sess = tf.Session(config=config) 
    
    SeqAug = returnAugmentationObj(0.)
    SeqAugTest = returnAugmentationObj(0.)

    filepath="YUV_LOG_preprocess.hdf5" # Name and path of weights to save

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') # Checkpoint call back to save best model on validation set

    lrd=ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1, mode='auto', min_delta=0.00001, cooldown=5, min_lr=0.00000000000000000001) # Callback to control learning rate on plateau condition 
    callbacks_list = [checkpoint,lrd]

    optimizer = Adam(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True) # Optimizer used to train
    

    
    if useTransferLearning:
        model = modelTransferLearning(True,"red_blue_transf.hdf5")
        model.compile(optimizer=optimizer, loss={"cat_kash":"binary_crossentropy","reg_kash":"mean_squared_error","reg_red":"mean_squared_error"}, metrics={"cat_kash":'accuracy',"reg_blue":"mse","reg_red":"mse"})
    elif useShred:
        model = modelShredding(False,"YUV_LOG_preprocess.hdf5")
        model.compile(optimizer=optimizer, loss={"classification":"categorical_crossentropy"},metrics={"classification":['acc',f1_m,precision_m, recall_m]})
    else:
        model = returnModel(False,"red_blue_cust.hdf5")
        model.compile(optimizer=optimizer, loss={"cat_kash":"binary_crossentropy","reg_kash":"mean_squared_error","reg_red":"mean_squared_error"},metrics={"cat_kash":'accuracy',"reg_blue":"mse","reg_red":"mse"})
        

    
    X,Y,name,name_faint,X_faint,Y_faint = loadData(800,0) # Load all data in one batch, 3 since sample data has only 3

    X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size=0.2, random_state=42) # Split data into training and testing 
    X_faint_tr,X_faint_te,Y_faint_tr,Y_faint_te =train_test_split(X_faint, Y_faint, test_size=0.2, random_state=42)
    
    
    [X_train.append(obj) for obj in X_faint_tr]
    [X_test.append(obj) for obj in X_faint_te]
    [y_train.append(obj) for obj in Y_faint_tr]
    [y_test.append(obj) for obj in Y_faint_te]
    [name.append(obj) for obj in name_faint]
#     X_train = np.concatenate((X_train,X_faint_tr))
#     X_test = np.concatenate((X_test,X_faint_te))
    
#     y_train = np.concatenate((y_train,Y_faint_tr))
#     y_test = np.concatenate((y_test,Y_faint_te))
    
 # Compile model for training


    for iterationOut in range(20): # increase for more iterations
        for iterationIn in range(10): # increase for more number of iterations
            xx_tr=[]
            yy_reg_tr=[]
            yy_cat_tr=[]
            xx_te=[]
            yy_reg_te=[]
            yy_cat_te=[]
            for i in range(4): # increase for more augmented data
                images_aug_tr, keypoints_aug_tr = SeqAug(images=X_train,keypoints=y_train)  
                tar_train_reg,tar_train_cat=key2Target(keypoints_aug_tr,name)
                images_aug_te, keypoints_aug_te = SeqAugTest(images=X_test,keypoints=y_test)
                tar_test_reg,tar_test_cat=key2Target(keypoints_aug_te,name)

                for ind,im in enumerate(images_aug_tr):
                    im=im[1000:1500,:,:] # Crop out only test area
                    #im = gaussBlur(im)
                    #im = gaborFilt(im)
                    #im = normalize(LOG(im,LOGker))
                    #im = enhanceImage(im)
                    if(useTransferLearning):
                        xx_tr.append(preprocess_input(im))
                    else:
                        #im = im/255.0
                        #im = np.array(im,dtype=np.float32)
                        yuv_im =im #cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)
                        xx_tr.append(yuv_im)
                for ii in tar_train_reg:
                    yy_reg_tr.append(ii)
                for ind,ii in enumerate(tar_train_cat):
                    if False: # Set to true to save train images augmentated
                        images_aug_tr[ind] = cv2.putText(images_aug_tr[ind],str(tar_train_cat[ind]),(0,20), font, 0.5,(255,0,0),2,cv2.LINE_AA)
                        images_aug_tr[ind] = cv2.circle(images_aug_tr[ind],(50,int(keypoints_aug_tr[ind][0][1])),5, (0,0,255), 5)
                        try:
                            images_aug_tr[ind] = cv2.circle(images_aug_tr[ind],(50,int(keypoints_aug_tr[ind][1][1])),5, (255,0,0), 5)
                        except:
                            pass
                        images_aug_tr[ind] = cv2.cvtColor(images_aug_tr[ind], cv2.COLOR_RGB2BGR)

                        cv2.imwrite("./sample/train/"+str(iterationOut)+str(iterationIn)+str(ind)+".jpg",images_aug_tr[ind][1000:1500,:,:])
                    yy_cat_tr.append(ii)
                for ind,im in enumerate(images_aug_te):
                    im=im[1000:1500,:,:] # Crop out only test area
                    #im = gaussBlur(im)
                    #im = gaborFilt(im)
                    #im = normalize(LOG(im,LOGker))
                    #im = enhanceImage(im)
                    if(useTransferLearning):
                        xx_te.append(preprocess_input(im))
                    else:
                        #im = im/255.0
                        #im = np.array(im,dtype=np.float32)
                        yuv_im =im #cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)
                        xx_te.append(yuv_im)
                for ii in tar_test_reg:
                    yy_reg_te.append(ii)
                for ind,ii in enumerate(tar_test_cat):
                    if False: # Set to true to save test images augmentated
                        images_aug_te[ind] = cv2.putText(images_aug_te[ind],str(tar_test_cat[ind]),(0,20), font, 0.5,(255,0,0),2,cv2.LINE_AA)
                        images_aug_te[ind] = cv2.circle(images_aug_te[ind],(50,int(keypoints_aug_te[ind][0][1])),5, (0,0,255), 5)
                        try:
                            images_aug_te[ind] = cv2.circle(images_aug_te[ind],(50,int(keypoints_aug_te[ind][1][1])),5, (255,0,0), 5)
                        except:
                            pass
                        images_aug_te[ind] = cv2.cvtColor(images_aug_te[ind], cv2.COLOR_RGB2BGR)

                        cv2.imwrite("./sample/test/"+str(iterationOut)+str(iterationIn)+str(ind)+".jpg",images_aug_te[ind])
                    yy_cat_te.append(ii)
            print("MAX",np.max(xx_tr),"MIN",np.min(xx_tr))
            xxx=np.array(xx_tr)
            yyy_reg=np.array(yy_reg_tr)
            yyy_cat=np.array(yy_cat_tr)
            xxx_te=np.array(xx_te)
            yyy_reg_te=np.array(yy_reg_te)
            yyy_cat_te=np.array(yy_cat_te)
            if useShred:
                SeqAug = returnAugmentationObj(0.8)
                SeqAugTest = returnAugmentationObj(0.1)
                xxx_train,yyyy_cat_tr=convertProblem2Shred(xxx,yyy_reg,SeqAug,LOGker)
#                 xxx_train = SeqAug(images=xxx_train)
#                 xxx_train = xxx_train/255.0
                xxx_test,yyyy_cat_te=convertProblem2Shred(xxx_te,yyy_reg_te,SeqAugTest,LOGker)
                print("MAX_tr",np.max(xxx_train),"MIN_tr",np.min(xxx_train))
                print("MAX_te",np.max(xxx_test),"MIN_te",np.min(xxx_test))
                if False:
                    for ind,img in enumerate(xxx_train):
                        tmpim = cv2.cvtColor(xxx_train[ind],cv2.COLOR_RGB2BGR)
                        tmpim=tmpim*255
                        tmpim = np.asarray(tmpim,dtype=np.uint8)
                        tmplab=list(yyyy_cat_tr[ind])
                        cv2.imwrite("./sample/train/"+str(ind)+"_"+str(tmplab.index(1))+".jpg",tmpim)
    
    
                training_generator = MixupGenerator(xxx_train, yyyy_cat_tr, batch_size=32, alpha=0.1)()
                class_weights = {0:2,1:4,2:1}
                #xxx_train=normalize(xxx_train)
                #xxx_test=normalize(xxx_test)
                model.fit(xxx_train, yyyy_cat_tr, validation_data=(xxx_test, yyyy_cat_te), epochs=150,batch_size=32,callbacks=callbacks_list)
#                 model.fit_generator(training_generator,validation_data=(xxx_test,yyyy_cat_te),epochs=5,steps_per_epoch=xxx_train.shape[0] // 32,callbacks=callbacks_list)
            else:
                model.fit(xxx, [yyy_cat,yyy_reg[:,0],yyy_reg[:,1]], validation_data=(xxx_te, [yyy_cat_te,yyy_reg_te[:,0],yyy_reg_te[:,1]]), epochs=35,batch_size=4,callbacks=callbacks_list) # Change batch size as per available resources
            
