import os
import time
import shutil
import numpy as np
import tensorflow as tf
from utils import data_loader
from core.model_new import ObjectDetection
from core.config import cfg
import utils.utils as utils
import tensorflow.keras.backend as K
import cv2
from tensorflow import keras
import tensorflow.keras.backend as K
# import tensorflow_addons as tfa
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


@tf.function
def custloss(y_true,y_pred):
    mask = y_true*y_pred
    common = tf.reduce_sum(mask,axis=[1,2,3])
    c_ground = tf.reduce_sum(y_true,axis=[1,2,3])
    c_predict = tf.reduce_sum(y_pred,axis=[1,2,3])
    l =1.0 - (1.0+common) /(1.0+c_ground+c_predict-common)

    print(l.shape)


    categorical_loss = tf.keras.losses.categorical_crossentropy(
                                                            y_true,
                                                            y_pred,
                                                            from_logits=False,
                                                            label_smoothing=0
                                                        )
    categorical_loss = tf.reduce_sum(categorical_loss,axis=[1,2])
    total_loss = categorical_loss + l

    return total_loss




@tf.function
def custlossSSD(y_true,y_pred):

    print(y_pred.shape,y_true.shape)

    categoricalLoss = tf.nn.sigmoid_cross_entropy_with_logits(y_true[:,:,:,8:],y_pred[:,:,:,8:])
    
    l1_smooth = tf.keras.losses.mean_squared_error(y_true[:,:,:,:8],tf.math.tanh(y_pred[:,:,:,:8]) )
    total_loss = tf.reduce_mean(categoricalLoss)+tf.reduce_mean(l1_smooth)
    return total_loss




@tf.function
def closs(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

class Train(object):
    def __init__(self):
        self.classes             = utils.read_class_names(cfg.RDT_Reader.CLASSES)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.saveModelpath       = cfg.TEST.WEIGHT_FILE
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.train_logdir        = "./dataset/log/"
        # self.trainset            = data_loader.loadDataObjSSDFromYoloFormat('train')
        # self.testset             = data_loader.loadDataObjSSDFromYoloFormat('test')
        print("*************LOADING DATA*************")
        self.filePathsTrain = data_loader.getImagePaths("train")
        self.labelDictTrain=data_loader.getAllAnnotations("train",self.filePathsTrain)
        self.filePathsTest = data_loader.getImagePaths("test")
        self.labelDictTest=data_loader.getAllAnnotations("test",self.filePathsTest)
        self.batchSizeTrain  = cfg.TRAIN.BATCH_SIZE
        self.batchSizeTest = cfg.TEST.BATCH_SIZE
        self.stepPerEpochTrain =int(len(self.filePathsTrain)/self.batchSizeTrain)-1     
        self.stepPerEpochTest = int(len(self.filePathsTest)/self.batchSizeTest)-1
        print(self.stepPerEpochTrain,self.stepPerEpochTest)
        self.trainGen = data_loader.image_generator(self.filePathsTrain,self.labelDictTrain,"train")
        self.testGen = data_loader.image_generator(self.filePathsTest,self.labelDictTest,"test")

        self.model = ObjectDetection(False ,self.initial_weight).model
        self.number_blocks = cfg.TRAIN.NUMBER_BLOCKS

    def train(self):

        checkpoint = keras.callbacks.ModelCheckpoint(self.saveModelpath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') # Checkpoint call back to save best model on validation set

        lrd=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=100, verbose=1, mode='auto', min_delta=0.00001, cooldown=5, min_lr=0.0000000001) # Callback to control learning rate on plateau condition 
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.train_logdir,update_freq="batch")

        callbacks_list = [checkpoint,lrd]

        optimizer = keras.optimizers.Adam(lr=0.00009, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True) # Optimizer used to train
        
        # X_train,y_train,name=trainset
        # X_test,y_test,name=testset
        self.model.compile(optimizer=optimizer,  # Optimizer
              # Loss function to minimize
              loss=custlossSSD)
        history = self.model.fit(self.trainGen,
                    epochs=1000,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=self.testGen,callbacks=callbacks_list, steps_per_epoch = self.stepPerEpochTrain,validation_steps = self.stepPerEpochTest )
        print('\nhistory dict:', history.history)




class Test(object):
    def __init__(self):
        K.clear_session()

        K.set_learning_phase(0)
        self.classes             = utils.read_class_names(cfg.RDT_Reader.CLASSES)
        self.num_classes         = len(self.classes)
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.TRAIN.MOVING_AVE_DECAY
        self.train_logdir        = "./dataset/log/train"
        # self.testset             = data_loader.loadDataObjSSD('test')
        self.checkpoint_name     = cfg.TEST.EVAL_MODEL_PATH+"/eval.ckpt"
        self.model_path          = cfg.TEST.EVAL_MODEL_PATH+"/model/"
        self.eval_tflite         = cfg.TEST.EVAL_MODEL_PATH+"/OD_180x320_HIV.lite"
        self.initial_weight      = cfg.TEST.WEIGHT_FILE
        self.output_node_names   = ["define_loss/reshapedOutput"]
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        # self.trainset            = data_loader.loadData('train')
        self.testset             = data_loader.loadDataObjSSDFromYoloFormat('test')
        # self.trainset             = data_loader.loadDataObjSSDFromYoloFormat('train')

        # self.steps_per_period    = len(self.trainset)
        self.quant_delay         = cfg.TRAIN.QUANT_DELAY
        self.quantizedPb         = cfg.TEST.QUANTIZED_WEIGHT_FILE
        self.resize_dim = tuple(cfg.TEST.INPUT_SIZE)
        self.number_blocks = cfg.TRAIN.NUMBER_BLOCKS


        self.model = ObjectDetection(True, self.initial_weight).model
        self.anch = cfg.TRAIN.ANCHOR_ASPECTRATIO  

            # self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
        

    def createTflite(self):
        
        print(self.model.summary())
        lite = tf.lite.TFLiteConverter.from_keras_model(self.model)
        lite.optimizations = [tf.lite.Optimize.DEFAULT]
        # lite.representative_dataset = self.representative_dataset_gen
        tflite_quant_model = lite.convert()
        open(self.eval_tflite, "wb").write(tflite_quant_model)

    def non_max_suppression_fast(self,boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes	
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")
    
    def representative_dataset_gen(self):
        for _ in range(len(self.testset[0])):
            inp,tr,na= self.testset
            img = inp[_]
            img = np.reshape(img,(1,self.resize_dim[0],self.resize_dim[1],1))

            # print(img.shape)
            yield [img]

    def runOntest(self):
        predictions=self.model.predict(self.testset[0])
        print(predictions.shape)
        resizefactor = int(self.resize_dim[0]/self.number_blocks)
        # for i,img in enumerate(self.testset[0]):
        
        for i,preds in enumerate(predictions):
            img = self.testset[0][i]
            img = img*255
            img = np.array(img,dtype=np.int32)
            print(img.shape)
            # print(preds[:,:,:,0:2]>0.8)
            box_0 =[]
            box_1 = []
            for row,p in enumerate (preds):
                for col,pp in enumerate(p):
                    for inind,anch in enumerate (pp):
                        # print(anch)
                        if anch[0]>0.95:
                            # print(row,col,inind,anch)

                            cy = (row-0.5)*resizefactor+anch[-4]*img.shape[0]
                            cx =  (col-0.5)*resizefactor+anch[-3]*img.shape[0]
                            w = self.anch[0][inind][0]+anch[-2]*img.shape[0]
                            h = self.anch[0][inind][1]+anch[-1]*img.shape[0]

                            x1,y1,x2,y2=data_loader.cxcy2xy([cx,cy,w,h])
                            box_0.append([x1,y1,x2,y2])
                            # img = cv2.rectangle(img,(int(y1),int(x1)),(int(y2),int(x2)),255,1) 
                        elif anch[1]>0.95:
                            # print(row,col,inind,anch)

                            cy = (row+0.5)*resizefactor+anch[-4]*img.shape[0]
                            cx =  (col+0.5)*resizefactor+anch[-3]*img.shape[0]
                            h = self.anch[0][inind][0]+anch[-2]*img.shape[0]
                            w = self.anch[0][inind][1]+anch[-1]*img.shape[0]

                            x1,y1,x2,y2=data_loader.cxcy2xy([cx,cy,w,h])
                            box_1.append([x1,y1,x2,y2])

                            # img = cv2.rectangle(img,(int(y1),int(x1)),(int(y2),int(x2)),0,1) 
            box_0 = np.array(box_0)
            box_1 = np.array(box_1)

            box_0=self.non_max_suppression_fast(box_0,0.7)
            box_1=self.non_max_suppression_fast(box_1,0.7)
            for b in box_0:
                img = cv2.rectangle(img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),255,1)
            for b in box_1:
                img = cv2.rectangle(img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),0,1)
            cv2.imwrite("output_check/"+str(i)+".jpg",img)
            # break

if __name__ == '__main__':
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    Train().train()
    # Test().createTflite()
    # Test().runOntest()



