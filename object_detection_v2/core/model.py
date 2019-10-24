import numpy as np
import tensorflow as tf
import utils.utils as utils
from utils import utils
from core.config import cfg
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_examples.models.pix2pix import pix2pix


class ObjectDetection(object):
    """Implement tensoflow model here"""
    def __init__(self, loadWeights, weightsFile):

        self.classes          = utils.read_class_names(cfg.RDT_Reader.CLASSES)
        self.num_class        = cfg.TRAIN.NUMBER_CLASSES
        self.loadWeights      = loadWeights
        self.weightsFile      = weightsFile
        self.resize_dim       = tuple(cfg.TEST.INPUT_SIZE) 
        self.number_anchors   = len(cfg.TRAIN.ANCHOR_ASPECTRATIO[0])

        self.model            = self.__build_network__imgClass()
    def __build_network__(self):   
        """This function returns a keras model.
            
            Args:
                loadWeights (bool) : Load weights specified in the weightsFile param
                weightsFile (str) : Path to weights
            
            Returns:
                :class:`keras.model.Model` : Neural Network 

        """
        x = keras.Input(shape=(self.resize_dim[1],self.resize_dim[0],3))

        conv1 = layers.Conv2D(128, (9,9), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(x)
        batchnorm1 = layers.BatchNormalization()(conv1)
        act1 = layers.LeakyReLU()(batchnorm1)
        # maxpool1 = layers.MaxPooling2D((2,2))(act1)


        conv2 = layers.Conv2D(64, (9,9), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(act1)
        batchnorm2 = layers.BatchNormalization()(conv2)
        act2 = layers.LeakyReLU()(batchnorm2)
        # maxpool2 = layers.MaxPooling2D((2,2))(act2)

        conv3 = layers.Conv2D(32, (9,9), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(act2)
        batchnorm3 = layers.BatchNormalization()(conv3)
        act3 = layers.LeakyReLU()(batchnorm3)
        # maxpool3 = layers.MaxPooling2D((2,2))(act3)

        out = tf.keras.layers.Conv2DTranspose(self.num_class+1, 9, padding='same', activation='softmax')(act3)
        
        # conv4 = layers.Conv2D(6, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(act3)
        # batchnorm4 = layers.BatchNormalization()(conv4)
        # act4 = layers.Conv2DTranspose('softmax',name="output")(batchnorm4)
        # maxpool4 = layers.MaxPooling2D((2,2))(act4)

        # flat1 = layers.Flatten()(x)
        # D1 = layers.Dense(128,kernel_initializer=keras.initializers.lecun_uniform(seed=None))(flat1)
        # batchnorm5 = layers.BatchNormalization()(D1)
        # act5 = layers.LeakyReLU()(batchnorm5)

        # D2 = layers.Dense(64,kernel_initializer=keras.initializers.lecun_uniform(seed=None))(act5)
        # batchnorm6 = layers.BatchNormalization()(D2)
        # act6 = layers.LeakyReLU()(batchnorm6)


        # D_reg = layers.Dense(self.num_class*2,kernel_initializer=keras.initializers.lecun_uniform(seed=None))(act6)
        # batchnorm7 = layers.BatchNormalization()(D_reg)
        # out = layers.Activation('linear',name="output")(batchnorm7)
        # outReshape = layers.Reshape((self.num_class, 2))(out)
        model = keras.Model(inputs=x, outputs=[out])
        if (self.loadWeights):
            model.load_weights(self.weightsFile,by_name=True)
        return model
    

    def __build_network__mobileNet(self):   
        """This function returns a keras model.
            
            Args:
                loadWeights (bool) : Load weights specified in the weightsFile param
                weightsFile (str) : Path to weights
            
            Returns:
                :class:`keras.model.Model` : Neural Network 

        """
        base_model = tf.keras.applications.MobileNetV2(input_shape=[self.resize_dim[1],self.resize_dim[0], 3], include_top=False)

        # Use the activations of these layers
        layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        # 'block_6_expand_relu',   # 16x16
        # 'block_13_expand_relu',  # 8x8
        # 'block_16_project',      # 4x4
        ]
        Layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=Layers)

        down_stack.trainable = True

        # print(down_stack.summary())
        inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        featurization = down_stack(inputs)

        # conv_inc_channels = layers.Conv2D(576, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(featurization[-1])
        # batchnorm1 = layers.BatchNormalization()(conv_inc_channels)
        # act1 = layers.LeakyReLU()(batchnorm1)

        # upsample_layer_1 = layers.Conv2DTranspose(576, 2, padding='valid',strides=2,activation='relu',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(act1)
        # concat_1 = layers.concatenate([featurization[-2],upsample_layer_1])

        # conv_dec_channels_2 = layers.Conv2D(192, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(concat_1)
        # batchnorm2 = layers.BatchNormalization()(conv_dec_channels_2)
        # act2 = layers.LeakyReLU()(batchnorm2)

        # upsample_layer_2 = layers.Conv2DTranspose(192, 2, padding='valid',strides=2,activation='relu',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(act2)
        # concat_2 = layers.concatenate([featurization[-3],upsample_layer_2])

        # conv_dec_channels_3 = layers.Conv2D(144, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(concat_2)
        # batchnorm3 = layers.BatchNormalization()(conv_dec_channels_3)
        # act3 = layers.LeakyReLU()(batchnorm3)

        # upsample_layer_3 = layers.Conv2DTranspose(144, 2, padding='valid',strides=2,activation='relu',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(act3)
        # concat_3 = layers.concatenate([featurization[-4],upsample_layer_3])

        conv_dec_channels_4 = layers.Conv2D(96, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(featurization[-1])
        batchnorm4 = layers.BatchNormalization()(conv_dec_channels_4)
        act4 = layers.LeakyReLU()(batchnorm4)

        upsample_layer_4 = layers.Conv2DTranspose(96, 2, padding='valid',strides=2,activation='relu',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(act4)
        concat_4 = layers.concatenate([featurization[-2],upsample_layer_4])

        conv_dec_channels_5 = layers.Conv2D(36, (3,3), padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(concat_4)
        batchnorm5 = layers.BatchNormalization()(conv_dec_channels_5)
        act5 = layers.LeakyReLU()(batchnorm5)

        upsample_layer_5 = layers.Conv2DTranspose(16, 2, padding='valid',strides=2,activation='relu',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(act5)
        
        conv_dec_channels_6 = layers.Conv2D(self.num_class+1,1, padding='same',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(upsample_layer_5)
        batchnorm6 = layers.BatchNormalization()(conv_dec_channels_6)
        act6 = layers.Softmax()(batchnorm6)

        print(upsample_layer_5.shape)
        
        model = keras.Model(inputs=inputs, outputs=[act6])
        if (self.loadWeights):
            model.load_weights(self.weightsFile,by_name=True)


        return model


    def __build_network__imgClass(self):   
            """This function returns a keras model.
                
                Args:
                    loadWeights (bool) : Load weights specified in the weightsFile param
                    weightsFile (str) : Path to weights
                
                Returns:
                    :class:`keras.model.Model` : Neural Network 

            """
            inputs = tf.keras.layers.Input(shape=[256, 256, 1])



            conv1 = keras.layers.Conv2D(8,(5,5), padding='valid',activation='relu',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(inputs)
            conv2 = keras.layers.Conv2D(16,(3,3), padding='valid',activation='relu',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(conv1)
            maxpool1 = keras.layers.MaxPooling2D((2,2))(conv2)
            drop1 = keras.layers.Dropout(0.2)(maxpool1)
            conv3 = keras.layers.Conv2D(32,(3,3), padding='valid',activation='relu',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(drop1)
            # conv4 = keras.layers.Conv2D(32,(3,3), padding='valid',activation='relu',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(conv3)
            maxpool2 = keras.layers.MaxPooling2D((4,4))(conv3)
            drop2 = keras.layers.Dropout(0.2)(maxpool2)
            
            conv5 = keras.layers.Conv2D(64,(1,1), padding='valid',activation='relu',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(drop2)
            maxpool3 = keras.layers.MaxPooling2D((2,2))(conv5)
            drop3 = keras.layers.Dropout(0.2)(maxpool3)
            conv6 = keras.layers.Conv2D(self.number_anchors*(self.num_class+4),(3,3), padding='same',activation='linear',kernel_initializer=keras.initializers.lecun_uniform(seed=None))(drop3)
            #drop2 = keras.layers.Dropout(0.2)(conv4)
            reshapeOut = layers.Reshape((15*15,self.number_anchors,(self.num_class+4)))(conv6)
            # for i in range(self.num_class+4):
            reshapeOutClass = reshapeOut[:,:,:,0:self.num_class]
            reshapeOutReg = reshapeOut[:,:,:,self.num_class:]
            reshapeOutClass = layers.Softmax()(reshapeOutClass)
            reshapeOutReg = layers.Activation("tanh")(reshapeOutReg)
            concat_1 = layers.concatenate([reshapeOutClass,reshapeOutReg])
            print(concat_1.shape)

            model = tf.keras.Model(inputs=inputs,outputs=[concat_1])

            if (self.loadWeights):
                model.load_weights(self.weightsFile,by_name=True)


            return model



