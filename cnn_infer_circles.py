# first version coded by Duo Wang (wd263@cam.ac.uk) 2018


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout, Activation, Flatten, RepeatVector,GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Lambda, GRU, LSTM
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.regularizers import l2
from keras.utils import plot_model
from keras.metrics import BinaryAccuracy
from keras import backend as K

import load_circle_infer as load_data
test_mode = True

nb_classes = 4
data_augmentation = True
nb_epoch=5
show_summary=True
plot_net=True
batch_size=128
if test_mode:
    data_path='euler_circle_generator/generated_diagram/Modus_Bamalip'
else:
    data_path='euler_circle_generator/generated_diagram'

model_dir='vaegan_models'      #'/local/scratch/wd263/vaegan_models/'
max_num_images=96000
depth_unit=64
regularizer=1e-5
hidden_dim=128
rnn_dim=128

dataset,labels=load_data.load(data_path,max_num_images)

# labels=labels[:,2:]
print('data shape:', dataset.shape)
print('label shape:',labels.shape)

if test_mode:
    test_dataset = dataset[:]
    test_labels = labels[:]
else:
    training_dataset=dataset[:80000]
    training_labels=labels[:80000]
    valid_dataset=dataset[80000:88000]
    valid_labels=labels[80000:88000]
    test_dataset=dataset[88000:]
    test_labels=labels[88000:]



if test_mode:
    model = load_model('euler_circle_model_duo.h5')
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(test_dataset, test_labels, batch_size=batch_size)
    print("test loss, test acc:", results)
else:

    inputAB = Input(shape=(64, 64, 6))
    inputA = Lambda(lambda x: x[:, :, :, 0:3], output_shape=(64, 64, 3))(inputAB)
    inputB = Lambda(lambda x: x[:, :, :, 3:6], output_shape=(64, 64, 3))(inputAB)

    shared_conv1 = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(regularizer))
    layerA_1 = shared_conv1(inputA)
    layerB_1 = shared_conv1(inputB)
    layerA_1 = BatchNormalization(axis=1)(layerA_1)
    layerB_1 = BatchNormalization(axis=1)(layerB_1)
    layerA_1 = Activation('relu')(layerA_1)
    layerB_1 = Activation('relu')(layerB_1)

    shared_conv2 = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(regularizer))
    layerA_2 = shared_conv2(layerA_1)
    layerB_2 = shared_conv2(layerB_1)
    layerA_2 = BatchNormalization(axis=1)(layerA_2)
    layerB_2 = BatchNormalization(axis=1)(layerB_2)
    layerA_2 = Activation('relu')(layerA_2)
    layerB_2 = Activation('relu')(layerB_2)
    layerA_2 = MaxPooling2D((2, 2), padding="same", strides=(2, 2))(layerA_2)
    layerB_2 = MaxPooling2D((2, 2), padding="same", strides=(2, 2))(layerB_2)

    shared_conv3 = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(regularizer))
    layerA_3 = shared_conv3(layerA_2)
    layerB_3 = shared_conv3(layerB_2)
    layerA_3 = BatchNormalization(axis=1)(layerA_3)
    layerB_3 = BatchNormalization(axis=1)(layerB_3)
    layerA_3 = Activation('relu')(layerA_3)
    layerB_3 = Activation('relu')(layerB_3)

    shared_conv4 = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(regularizer))
    layerA_4 = shared_conv4(layerA_3)
    layerB_4 = shared_conv4(layerB_3)
    layerA_4 = BatchNormalization(axis=1)(layerA_4)
    layerB_4 = BatchNormalization(axis=1)(layerB_4)
    layerA_4 = Activation('relu')(layerA_4)
    layerB_4 = Activation('relu')(layerB_4)

    layerA_4 = MaxPooling2D((2, 2), padding="same", strides=(2, 2))(layerA_4)
    layerB_4 = MaxPooling2D((2, 2), padding="same", strides=(2, 2))(layerB_4)

    shared_conv5 = Conv2D(32, (3, 3), padding="valid", kernel_regularizer=l2(regularizer))
    layerA_5 = shared_conv5(layerA_4)
    layerB_5 = shared_conv5(layerB_4)
    layerA_5 = BatchNormalization(axis=1)(layerA_5)
    layerB_5 = BatchNormalization(axis=1)(layerB_5)
    layerA_5 = Activation('relu')(layerA_5)
    layerB_5 = Activation('relu')(layerB_5)

    shared_conv6 = Conv2D(32, (3, 3), padding="valid", kernel_regularizer=l2(regularizer))
    layerA_6 = shared_conv6(layerA_5)
    layerB_6 = shared_conv6(layerB_5)
    layerA_6 = BatchNormalization(axis=1)(layerA_6)
    layerB_6 = BatchNormalization(axis=1)(layerB_6)
    layerA_6 = Activation('relu')(layerA_6)
    layerB_6 = Activation('relu')(layerB_6)

    layerA_6 = MaxPooling2D((2, 2), padding="same", strides=(2, 2))(layerA_6)
    layerB_6 = MaxPooling2D((2, 2), padding="same", strides=(2, 2))(layerB_6)

    shared_conv7 = Conv2D(32, (3, 3), padding="valid", kernel_regularizer=l2(regularizer))
    layerA_7 = shared_conv7(layerA_6)
    layerB_7 = shared_conv7(layerB_6)
    layerA_7 = BatchNormalization(axis=1)(layerA_7)
    layerB_7 = BatchNormalization(axis=1)(layerB_7)
    layerA_7 = Activation('relu')(layerA_7)
    layerB_7 = Activation('relu')(layerB_7)

    concatenate_map = concatenate([layerA_7, layerB_7])

    model_cnn = Model(inputs=inputAB, outputs=concatenate_map)
    model_cnn.summary()

    model_cnn.save(model_dir + 'euler_cnn_k2_untrained.h5')
    plot_model(model_cnn, to_file='euler_cnn.png')
    input_image = Input(shape=(64, 64, 6))
    feature_map = model_cnn(input_image)
    flattened = Flatten()(feature_map)
    # avg_pool=GlobalAveragePooling2D()(feature_map)

    fc = Dense(128, kernel_regularizer=l2(regularizer))(flattened)
    fc = BatchNormalization()(fc)
    fc = Activation('relu')(fc)

    fc = Dropout(0.3)(fc)

    fc = Dense(64, kernel_regularizer=l2(regularizer))(flattened)
    fc = BatchNormalization()(fc)
    fc = Activation('relu')(fc)

    fc = Dropout(0.3)(fc)

    out = Dense(nb_classes, kernel_regularizer=l2(regularizer))(fc)
    out = Activation('sigmoid')(out)
    
    model = Model(inputs=input_image, outputs=out)
    model_fc = Model(inputs=input_image, outputs=fc)
    if show_summary:
        print(model.summary())

    adam=Adam(lr=0.0002, beta_1=0.9, beta_2=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    # Evaluate the model on the test data using `evaluate`


    # def binary_accuracy(y_true, y_pred):
    #    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

    # pred_labels = model.predict(test_dataset, batch_size = batch_size)
    # result = binary_accuracy(pred_labels, test_labels).numpy()
    # print("test loss, test acc:", result)

    ## results = model.evaluate(test_dataset, test_labels, batch_size=batch_size)
    ## print("test loss, test acc:", results)


    if not data_augmentation:
        model.fit(training_dataset, training_labels, epochs=nb_epoch,
                  batch_size=batch_size,validation_data=(valid_dataset, valid_labels),
                  verbose=0)
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images


        # fit the model on the batches generated by datagen.flow()
        history_log=model.fit_generator(datagen.flow(training_dataset, training_labels,
                            batch_size=batch_size),
                            samples_per_epoch=training_dataset.shape[0],
                            epochs=nb_epoch,
                            validation_data=(valid_dataset, valid_labels))
        model.save('euler_circle_model5.h5')
        model_fc.save('euler_circle_fc5.h5')




