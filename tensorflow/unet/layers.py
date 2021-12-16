import tensorflow as tf


def double_conv(inputs,filters):
    
    #first conv
    conv = tf.keras.layers.Conv2D(filters)(inputs)
    conv = tf.keras.layers.Activation('relu')(conv)
    
    #second conv
    conv = tf.keras.layers.Conv2D(filters)(conv)
    conv = tf.keras.layers.Activation('relu')(conv)
    
    return conv
    

def unet_encoder(inputs, filters):

    #double conv output
    conv = double_conv(inputs,filters)

    #pooling
    pool = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv)

    return conv, pool

def unet_decoder(inputs,filters,previous_conv):
    
    deconv = tf.keras.layers.Conv2DTranspose(filters)(inputs)
    concat = tf.concat([previous_conv,deconv])
    deconv = double_conv(concat, filters)
    
    return deconv


def unet(inputs, filters):

    #Encoder
    conv1, pool1 = unet_encoder(tf.keras.Input(inputs), filters)
    conv2, pool2 = unet_encoder(pool1, filters*2)
    conv3, pool3 = unet_encoder(pool2, filters*4)
    conv4, pool4 = unet_encoder(pool3, filters*8)

    #Bottleneck 
    bn, _ = unet_encoder(pool4, filters*16)

    #Decoder
    deconv1 = unet_decoder(bn, filters*8, conv4)
    deconv2 = unet_decoder(deconv1, filters*4, conv3)
    deconv3 = unet_decoder(deconv2, filters*2, conv2)
    deconv4 = unet_decoder(deconv3, filters, conv1)


    output = tf.keras.laters.Conv2D(1, activation = 'sigmoid')(deconv4)
    model = tf.keras.Model(inputs = tf.keras.Input(inputs), output = output)

    return model 
