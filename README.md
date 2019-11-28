# EIP4-SESSION4
## Final Validation accuracy for Base Network 81.66

## My Model
model = Sequential()
model.add(SeparableConv2D(filters=48, kernel_size=(3, 3), padding='same',  depth_multiplier=1, input_shape=(32, 32, 3)))
model.add(Dropout(0.07))
model.add(BatchNormalization())

model.add(Activation('relu'))
model.add(SeparableConv2D(filters=48,  kernel_size=(3, 3),padding='same',  depth_multiplier=1))
model.add(Dropout(0.07))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(SeparableConv2D(filters=96,  kernel_size=(3, 3), padding='same',  depth_multiplier=1))
model.add(Dropout(0.07))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(filters=96,  kernel_size=(3, 3),padding='same',  depth_multiplier=1))
model.add(Dropout(0.07))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(SeparableConv2D(filters=192,  kernel_size=(3, 3), padding='same',  depth_multiplier=1))
model.add(Dropout(0.07))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(filters=192,  kernel_size=(3, 3),padding='same',  depth_multiplier=1))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.07))

model.add(Flatten())
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## summery

Model: "sequential_13"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
separable_conv2d_80 (Separab (None, 32, 32, 48)        219       
_________________________________________________________________
dropout_110 (Dropout)        (None, 32, 32, 48)        0         
_________________________________________________________________
batch_normalization_68 (Batc (None, 32, 32, 48)        192       
_________________________________________________________________
activation_79 (Activation)   (None, 32, 32, 48)        0         
_________________________________________________________________
separable_conv2d_81 (Separab (None, 32, 32, 48)        2784      
_________________________________________________________________
dropout_111 (Dropout)        (None, 32, 32, 48)        0         
_________________________________________________________________
batch_normalization_69 (Batc (None, 32, 32, 48)        192       
_________________________________________________________________
activation_80 (Activation)   (None, 32, 32, 48)        0         
_________________________________________________________________
max_pooling2d_34 (MaxPooling (None, 16, 16, 48)        0         
_________________________________________________________________
dropout_112 (Dropout)        (None, 16, 16, 48)        0         
_________________________________________________________________
separable_conv2d_82 (Separab (None, 16, 16, 96)        5136      
_________________________________________________________________
dropout_113 (Dropout)        (None, 16, 16, 96)        0         
_________________________________________________________________
batch_normalization_70 (Batc (None, 16, 16, 96)        384       
_________________________________________________________________
activation_81 (Activation)   (None, 16, 16, 96)        0         
_________________________________________________________________
separable_conv2d_83 (Separab (None, 16, 16, 96)        10176     
_________________________________________________________________
dropout_114 (Dropout)        (None, 16, 16, 96)        0         
_________________________________________________________________
batch_normalization_71 (Batc (None, 16, 16, 96)        384       
_________________________________________________________________
activation_82 (Activation)   (None, 16, 16, 96)        0         
_________________________________________________________________
max_pooling2d_35 (MaxPooling (None, 8, 8, 96)          0         
_________________________________________________________________
dropout_115 (Dropout)        (None, 8, 8, 96)          0         
_________________________________________________________________
separable_conv2d_84 (Separab (None, 8, 8, 192)         19488     
_________________________________________________________________
dropout_116 (Dropout)        (None, 8, 8, 192)         0         
_________________________________________________________________
batch_normalization_72 (Batc (None, 8, 8, 192)         768       
_________________________________________________________________
activation_83 (Activation)   (None, 8, 8, 192)         0         
_________________________________________________________________
separable_conv2d_85 (Separab (None, 8, 8, 192)         38784     
_________________________________________________________________
dropout_117 (Dropout)        (None, 8, 8, 192)         0         
_________________________________________________________________
batch_normalization_73 (Batc (None, 8, 8, 192)         768       
_________________________________________________________________
activation_84 (Activation)   (None, 8, 8, 192)         0         
_________________________________________________________________
max_pooling2d_36 (MaxPooling (None, 4, 4, 192)         0         
_________________________________________________________________
dropout_118 (Dropout)        (None, 4, 4, 192)         0         
_________________________________________________________________
flatten_12 (Flatten)         (None, 3072)              0         
_________________________________________________________________
activation_85 (Activation)   (None, 3072)              0         
_________________________________________________________________
dropout_119 (Dropout)        (None, 3072)              0         
_________________________________________________________________
dense_8 (Dense)              (None, 10)                30730     
=================================================================
Total params: 110,005
Trainable params: 108,661
Non-trainable params: 1,344
_________________________________________________________________


## 50 epoch logs
