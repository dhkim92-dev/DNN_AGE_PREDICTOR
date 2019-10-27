import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Activation, Dense, \
GlobalAveragePooling2D,ZeroPadding2D, Add
import numpy as np

from tensorflow.keras.applications.resnet50 import ResNet50

#resnet = ResNet50()
#resnet.summary()

def resnetConv2D(x, filters=64, strides = (1,1), filters_scale = 1) :
    filters = filters*filters_scale

    x = Conv2D(filters,(1,1),strides=strides,padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(4*filters, (1, 1), strides=(1, 1), padding='valid')(x)

    return x

def resnetConv1(x) :
    x = ZeroPadding2D(padding=(3,3))(x)
    x = Conv2D(64,(7,7),strides=(2,2),padding = 'valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)

    return x

def resnetConv2(x) :
    x = MaxPooling2D((3,3), 2)(x)

    sc = x ## shortcut

    for i in range(0,3) :
        if i == 0 :
            x = resnetConv2D(x, strides = (1,1),filters_scale = 1)
            
            sc = Conv2D(256, kernel_size = (1, 1), strides=(1, 1), padding='valid')(sc)            
            x = BatchNormalization()(x)
            sc = BatchNormalization()(sc)
            
            x = Add()([x, sc])
            x = Activation('relu')(x)

            sc = x

        else : 
            x = resnetConv2D(x,strides = (1,1), filters_scale = 1)
            x = BatchNormalization()(x)

            x = Add()([x, sc])
            x = Activation('relu')(x)

            sc = x

    return x

def resnetConv3(x) :
    sc = x

    for i in range(0,4) : 
        if i == 0 :
            x = resnetConv2D(x,strides=(2,2),filters_scale=2)
            sc = Conv2D(512,kernel_size =(1,1),strides=(2,2),padding='valid')(sc)
            x=BatchNormalization()(x)
            sc = BatchNormalization()(sc)
            x = Add()([x,sc])
            x = Activation('relu')(x)
            sc = x
        
        else :
            x = resnetConv2D(x,strides = (1,1), filters_scale=2)
            x = BatchNormalization()(x)

            x = Add()([x,sc])
            x = Activation('relu')(x)

            sc = x

    return x

def resnetConv4(x) :
    sc = x

    for i in range(0, 6) :
        if i == 0 :
            x = resnetConv2D(x, strides = (2,2), filters_scale = 4)

            sc = Conv2D(filters = 1024, kernel_size = (1,1), strides = (2,2), padding = 'valid')(sc)
            x = BatchNormalization()(x)
            sc = BatchNormalization()(sc)
            x = Add()([x, sc])
            Activation('relu')(x)
            sc = x


        else :
            x = resnetConv2D(x, strides = (1,1), filters_scale=4)
            x = BatchNormalization()(x)

            x = Add()([x,sc])
            x = Activation('relu')(x)

            sc = x

    return x

def resnetConv5(x) :
    sc = x

    for i in range(0, 3) :
        if i == 0 :
            x = resnetConv2D(x,strides = (2,2),filters_scale = 8)

            sc = Conv2D(filters = 2048, kernel_size = (1,1), strides = (2,2),padding = 'valid')(sc)
            x = BatchNormalization()(x)
            sc = BatchNormalization()(sc)

            x = Add()([x,sc])
            x = Activation('relu')(x)
            sc = x
        else :
            x = resnetConv2D(x, strides = (1,1),filters_scale = 8)
            x = BatchNormalization()(x)

            x = Add()([x, sc])
            x = Activation('relu')(x)

            sc = x

    return x

def myModel() :
    _input = Input(shape = (128,128,3), dtype = 'float32', name='input')
    x = resnetConv1(_input)
    x = resnetConv2(x)
    x = resnetConv3(x)
    x = resnetConv4(x)
    x = resnetConv5(x)
    x = GlobalAveragePooling2D()(x)
    
    feature_vector = Dense(256, activation = 'relu',name = 'feature_vector')(x) ## feature vector

    output = Dense(1, activation='relu')(feature_vector)## age

    my_resnet_model = Model(_input,output)

    return my_resnet_model

#m = myResnet()
#m.summary()
