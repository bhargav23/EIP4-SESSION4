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
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  if sys.path[0] == '':
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, steps_per_epoch=390, epochs=50)`
  if sys.path[0] == '':
Epoch 1/50
390/390 [==============================] - 16s 40ms/step - loss: 1.7003 - acc: 0.4081 - val_loss: 1.5923 - val_acc: 0.4354
Epoch 2/50
390/390 [==============================] - 9s 24ms/step - loss: 1.1982 - acc: 0.5755 - val_loss: 1.1878 - val_acc: 0.5866
Epoch 3/50
390/390 [==============================] - 10s 24ms/step - loss: 1.0160 - acc: 0.6420 - val_loss: 1.0232 - val_acc: 0.6358
Epoch 4/50
390/390 [==============================] - 9s 24ms/step - loss: 0.8972 - acc: 0.6831 - val_loss: 0.8812 - val_acc: 0.6939
Epoch 5/50
390/390 [==============================] - 9s 24ms/step - loss: 0.8236 - acc: 0.7096 - val_loss: 0.8935 - val_acc: 0.6945
Epoch 6/50
390/390 [==============================] - 9s 24ms/step - loss: 0.7649 - acc: 0.7312 - val_loss: 0.9055 - val_acc: 0.6839
Epoch 7/50
390/390 [==============================] - 9s 24ms/step - loss: 0.7169 - acc: 0.7492 - val_loss: 0.8729 - val_acc: 0.6997
Epoch 8/50
390/390 [==============================] - 10s 24ms/step - loss: 0.6826 - acc: 0.7602 - val_loss: 0.7896 - val_acc: 0.7293
Epoch 9/50
390/390 [==============================] - 9s 24ms/step - loss: 0.6441 - acc: 0.7729 - val_loss: 0.7393 - val_acc: 0.7447
Epoch 10/50
390/390 [==============================] - 9s 24ms/step - loss: 0.6139 - acc: 0.7835 - val_loss: 0.7020 - val_acc: 0.7575
Epoch 11/50
390/390 [==============================] - 9s 24ms/step - loss: 0.5871 - acc: 0.7948 - val_loss: 0.8161 - val_acc: 0.7288
Epoch 12/50
390/390 [==============================] - 9s 24ms/step - loss: 0.5663 - acc: 0.8019 - val_loss: 0.7448 - val_acc: 0.7467
Epoch 13/50
390/390 [==============================] - 10s 24ms/step - loss: 0.5440 - acc: 0.8078 - val_loss: 0.7521 - val_acc: 0.7379
Epoch 14/50
390/390 [==============================] - 9s 24ms/step - loss: 0.5246 - acc: 0.8142 - val_loss: 0.6847 - val_acc: 0.7617
Epoch 15/50
390/390 [==============================] - 9s 24ms/step - loss: 0.5057 - acc: 0.8220 - val_loss: 0.7096 - val_acc: 0.7590
Epoch 16/50
390/390 [==============================] - 9s 24ms/step - loss: 0.4850 - acc: 0.8291 - val_loss: 0.6954 - val_acc: 0.7613
Epoch 17/50
390/390 [==============================] - 9s 24ms/step - loss: 0.4765 - acc: 0.8341 - val_loss: 0.6165 - val_acc: 0.7907
Epoch 18/50
390/390 [==============================] - 10s 24ms/step - loss: 0.4622 - acc: 0.8368 - val_loss: 0.7617 - val_acc: 0.7392
Epoch 19/50
390/390 [==============================] - 10s 24ms/step - loss: 0.4471 - acc: 0.8407 - val_loss: 0.6467 - val_acc: 0.7770
Epoch 20/50
390/390 [==============================] - 9s 24ms/step - loss: 0.4383 - acc: 0.8459 - val_loss: 0.7440 - val_acc: 0.7626
Epoch 21/50
390/390 [==============================] - 9s 24ms/step - loss: 0.4225 - acc: 0.8501 - val_loss: 0.6573 - val_acc: 0.7809
Epoch 22/50
390/390 [==============================] - 9s 24ms/step - loss: 0.4118 - acc: 0.8551 - val_loss: 0.6254 - val_acc: 0.7929
Epoch 23/50
390/390 [==============================] - 9s 24ms/step - loss: 0.3991 - acc: 0.8571 - val_loss: 0.6174 - val_acc: 0.7903
Epoch 24/50
390/390 [==============================] - 9s 24ms/step - loss: 0.3938 - acc: 0.8615 - val_loss: 0.5955 - val_acc: 0.7967
Epoch 25/50
390/390 [==============================] - 10s 24ms/step - loss: 0.3838 - acc: 0.8642 - val_loss: 0.6276 - val_acc: 0.7909
Epoch 26/50
390/390 [==============================] - 9s 24ms/step - loss: 0.3777 - acc: 0.8657 - val_loss: 0.6793 - val_acc: 0.7849
Epoch 27/50
390/390 [==============================] - 10s 24ms/step - loss: 0.3650 - acc: 0.8694 - val_loss: 0.6148 - val_acc: 0.7998
Epoch 28/50
390/390 [==============================] - 9s 24ms/step - loss: 0.3574 - acc: 0.8723 - val_loss: 0.7016 - val_acc: 0.7704
Epoch 29/50
390/390 [==============================] - 9s 24ms/step - loss: 0.3502 - acc: 0.8754 - val_loss: 0.6958 - val_acc: 0.7698
Epoch 30/50
390/390 [==============================] - 9s 24ms/step - loss: 0.3470 - acc: 0.8757 - val_loss: 0.5987 - val_acc: 0.8048
Epoch 31/50
390/390 [==============================] - 9s 24ms/step - loss: 0.3439 - acc: 0.8760 - val_loss: 0.5744 - val_acc: 0.8163
Epoch 32/50
390/390 [==============================] - 9s 24ms/step - loss: 0.3337 - acc: 0.8798 - val_loss: 0.6521 - val_acc: 0.7887
Epoch 33/50
390/390 [==============================] - 9s 24ms/step - loss: 0.3275 - acc: 0.8823 - val_loss: 0.5649 - val_acc: 0.8104
Epoch 34/50
390/390 [==============================] - 10s 24ms/step - loss: 0.3172 - acc: 0.8870 - val_loss: 0.7668 - val_acc: 0.7670
Epoch 35/50
390/390 [==============================] - 9s 24ms/step - loss: 0.3184 - acc: 0.8851 - val_loss: 0.5999 - val_acc: 0.8057
Epoch 36/50
390/390 [==============================] - 9s 24ms/step - loss: 0.3018 - acc: 0.8917 - val_loss: 0.6320 - val_acc: 0.7995
Epoch 37/50
390/390 [==============================] - 9s 24ms/step - loss: 0.3024 - acc: 0.8910 - val_loss: 0.5732 - val_acc: 0.8188
Epoch 38/50
390/390 [==============================] - 9s 24ms/step - loss: 0.2999 - acc: 0.8910 - val_loss: 0.5529 - val_acc: 0.8226
Epoch 39/50
390/390 [==============================] - 9s 24ms/step - loss: 0.2994 - acc: 0.8921 - val_loss: 0.6498 - val_acc: 0.8035
Epoch 40/50
390/390 [==============================] - 9s 24ms/step - loss: 0.2881 - acc: 0.8963 - val_loss: 0.7583 - val_acc: 0.7657
Epoch 41/50
390/390 [==============================] - 10s 24ms/step - loss: 0.2873 - acc: 0.8969 - val_loss: 0.6014 - val_acc: 0.8119
Epoch 42/50
390/390 [==============================] - 9s 24ms/step - loss: 0.2816 - acc: 0.8985 - val_loss: 0.6578 - val_acc: 0.7964
Epoch 43/50
390/390 [==============================] - 9s 24ms/step - loss: 0.2844 - acc: 0.8974 - val_loss: 0.7218 - val_acc: 0.7861
Epoch 44/50
390/390 [==============================] - 9s 24ms/step - loss: 0.2761 - acc: 0.9013 - val_loss: 0.5822 - val_acc: 0.8155
Epoch 45/50
390/390 [==============================] - 9s 24ms/step - loss: 0.2737 - acc: 0.9027 - val_loss: 0.6682 - val_acc: 0.7895
Epoch 46/50
390/390 [==============================] - 9s 24ms/step - loss: 0.2693 - acc: 0.9025 - val_loss: 0.6258 - val_acc: 0.8073
Epoch 47/50
390/390 [==============================] - 9s 24ms/step - loss: 0.2644 - acc: 0.9036 - val_loss: 0.6347 - val_acc: 0.8017
Epoch 48/50
390/390 [==============================] - 9s 24ms/step - loss: 0.2605 - acc: 0.9058 - val_loss: 0.6465 - val_acc: 0.7995
Epoch 49/50
390/390 [==============================] - 9s 24ms/step - loss: 0.2609 - acc: 0.9066 - val_loss: 0.5896 - val_acc: 0.8151
Epoch 50/50
390/390 [==============================] - 9s 24ms/step - loss: 0.2561 - acc: 0.9074 - val_loss: 0.7128 - val_acc: 0.7976
Model took 480.62 seconds to train

Accuracy on test data is: 79.76
