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
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage
from sklearn.utils import class_weight
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda, AvgPool2D
from keras.models import Sequential,Model
from keras.layers import ReLU, Dense, Conv2D, Flatten,Dropout, MaxPooling2D, GlobalAveragePooling2D, LeakyReLU, Activation, BatchNormalization, Input, merge, Softmax
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import random
from keras.constraints import max_norm
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3,preprocess_input
import argparse
parser = argparse.ArgumentParser(description='Train a CNN to detect presence of red and blue line on an RDT and also give the normalized y-axis location.')
parser.add_argument('--transfer', default="false",
                    help='Set boolean to true to train a transfer learning model',type=str)
font = cv2.FONT_HERSHEY_SIMPLEX
args=parser.parse_args()

if args.transfer =="true" or args.transfer =="True":
    useTransferLearning = True
else:
    useTransferLearning = False

if useTransferLearning:
    print("\n\nCurrently using transfer learning...\n\n")
else:
    print("\n\nCurrently using custom model...\n\n")

def loadData(noBatchSamples,batchIdx,duplicativeFactor=30,rareData="redonly.txt",rootPathCentreLabel="./obj/labels",rootPathCroppedImages = "./obj/images"):
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
    name=[]
    test_data=[]
    f = open(rareData)
    lines = f.readlines()
    lines = [x.strip() for x in lines]
    f.close()
    for ind,element in enumerate(os.listdir(rootPathCentreLabel)):
        if  ind >= noBatchSamples*batchIdx and ind<=noBatchSamples*(batchIdx+1):
            with open(os.path.join(rootPathCentreLabel,element)) as fin:
                y = [(0,0),(0,0)]
                img = cv2.imread(os.path.join(rootPathCroppedImages,element.replace(".txt",".jpg")),cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for line in fin:
                    line=line.split(" ")
                    if line[0]=="0" :
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
                for i in range(duplicativeFactor):
                    y_train.append(y)
                    X_train.append(img)
                    name.append(element)
            else:
                y_train.append(y)
                X_train.append(img)
                name.append(element)
        else:
            test_data.append(element)
    with open("test_data.txt","w") as fout:
        fout.write("\n".join(test_data))
    X_train=np.array(X_train)  
    print("Training + validation:",len(X_train))
    return X_train,y_train,name #np.zeros((128, 32, 32, 3), dtype=np.uint8) + (batch_idx % 255)


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
        
        elif k[0][1]>=700 and (k[1][1]<=700 or k[1][1]>=1800): # Red line:False && Blue line:True
            y[0]=k[0][1]/2000
            y[1]=0
            y_class[0]=1
            y_class[1]=0
        elif (k[1][1]>=700 and k[1][1]<=1900) and k[0][1]>=700: # Red line:True && Blue line:True
            numred+=1
            y[0]=k[0][1]/2000
            y[1]=k[1][1]/2000
            y_class[0]=1
            y_class[1]=1
        y_test_regression.append(np.array(y))
        y_test_categorical.append(np.array(y_class))
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
            sometimes(iaa.Affine(
                translate_percent={"x": (-0.06, 0.06), "y": (-0.04, 0.04)}, # translate by -x to +x percent (per axis)
                rotate=(-5, 5) # rotate by -x to +x degrees
            )),
            # execute 0 to 2 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 2),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 4.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 8)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 13)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    iaa.Add((-9, 9), per_channel=0.5), # change brightness of images (by -x to x of original value)
                    iaa.AddToHueAndSaturation((-15, 15)), # change hue and saturation
                    iaa.GammaContrast((0.2,1.8)),
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.016))) # Add perscpective transform
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

    D_sigmoid = Dense(2)(act6)
    batchnorm8 = BatchNormalization()(D_sigmoid)
    out2 = Activation('sigmoid',name="reg_kash")(batchnorm8)

    model = Model(inputs=x, outputs=[out1,out2])
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




if __name__ == "__main__":
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) # Correctly put number of GPU and CPU
    sess = tf.Session(config=config) 
    
    SeqAug = returnAugmentationObj()
    if useTransferLearning:
        model = modelTransferLearning(True,"red_blue_transf.hdf5")
    else:
        model = returnModel(True,"red_blue_cust.hdf5")
        
    filepath="weights-latest_model_YCrCb_test.hdf5" # Name and path of weights to save
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') # Checkpoint call back to save best model on validation set

    lrd=ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1, mode='auto', min_delta=0.00001, cooldown=5, min_lr=0.00000000000000000001) # Callback to control learning rate on plateau condition 
    
    X,Y,name = loadData(540,0) # Load all data in one batch, 3 since sample data has only 3

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33333, random_state=42) # Split data into training and testing 

    callbacks_list = [checkpoint,lrd]

    optimizer = Adam(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True) # Optimizer used to train
    
    model.compile(optimizer=optimizer, loss={"cat_kash":"binary_crossentropy","reg_kash":lossReg}, metrics={"cat_kash":'accuracy',"reg_kash":"mse"}) # Compile model for training


    for iterationOut in range(5): # increase for more iterations
        for iterationIn in range(5): # increase for more number of iterations
            xx_tr=[]
            yy_reg_tr=[]
            yy_cat_tr=[]
            xx_te=[]
            yy_reg_te=[]
            yy_cat_te=[]
            for i in range(5): # increase for more augmented data
                images_aug_tr, keypoints_aug_tr = SeqAug(images=X_train,keypoints=y_train)  
                images_aug_te, keypoints_aug_te = SeqAug(images=X_test,keypoints=y_test)
                tar_train_reg,tar_train_cat=key2Target(keypoints_aug_tr,name)
                tar_test_reg,tar_test_cat=key2Target(keypoints_aug_te,name)
                for ind,im in enumerate(images_aug_tr):
                    im=im[1000:1500,:,:] # Crop out only test area
                    if(useTransferLearning):
                        xx_tr.append(preprocess_input(im))
                    else:
                        im = im/255.0
                        im = np.array(im,dtype=np.float32)
                        yuv_im = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)
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
                    if(useTransferLearning):
                        xx_te.append(preprocess_input(im))
                    else:
                        im = im/255.0
                        im = np.array(im,dtype=np.float32)
                        yuv_im = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)
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
            xxx=np.array(xx_tr)
            yyy_reg=np.array(yy_reg_tr)
            yyy_cat=np.array(yy_cat_tr)
            xxx_te=np.array(xx_te)
            yyy_reg_te=np.array(yy_reg_te)
            yyy_cat_te=np.array(yy_cat_te)
            model.fit(xxx, [yyy_cat,yyy_reg], validation_data=(xxx_te, [yyy_cat_te,yyy_reg_te]), epochs=35,batch_size=4,callbacks=callbacks_list) # Change batch size as per available resources
            
