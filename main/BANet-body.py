import numpy as np
import scipy.io
import h5py
import keras
import os
from keras.layers import *
from keras.layers.core import *
from keras.models import *
from keras.regularizers import *
from keras import metrics
from keras import backend as K
from keras.backend import sum, mean, max
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras_self_attention import SeqSelfAttention


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(2)



#-----------------------------------------||||||||||||||||||||||||||||||||||||||||||||||||||

# Reference:

# "Learning Bodily and Temporal Attention in Protective Movement Behavior Detection"
#  arxiv preprint arxiv:1904.10824 (2019)

# "Automatic Detection of Protective Behavior in Chronic Pain Physical Rehabilitation: A Recurrent Neural Network Approach."
#  arXiv preprint arXiv:1902.08990 (2019).

# If you find the code useful, please cite the paper above
# : )

#-----------------------------------------||||||||||||||||||||||||||||||||||||||||||||||||||



def loadata(testname, num, type):   # loading training and validation data, can be adjusted to your environment.
    data = scipy.io.loadmat('train' + num + '.mat')
    testdata = scipy.io.loadmat(testname + num + '.mat')
    testlabel = scipy.io.loadmat(testname + num + 'label.mat')
    X_train0 = data['data']
    y_train = data['label']
    X_valid0 = testdata['subdata']
    y_valid = testlabel['sublabel']
    num_classes = 2  # classes
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    _, win_len, dim = X_train0.shape
    return X_train0, X_valid0, y_train, y_valid


def crop(dimension, start, end):
    # Thanks to the nice person named marc-moreaux on Github page:https://github.com/keras-team/keras/issues/890 who created this beautiful and sufficient function: )
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)


def build_model():
    timestep = 180   # length of an input frame
    dimension = 30   # dimension of an input frame
    BodyNum = 13     # number of body segments (different sensors) to consider


    #Model 1: Temporal Information encoding model (keras Model API)
    singleinput = Input(shape=(180, 2,))
    lstm_units = 8
    LSTM1 = LSTM(lstm_units, return_sequences=True, implementation=1)(singleinput)
    Dropout1 = Dropout(0.5)(LSTM1)
    LSTM2 = LSTM(lstm_units, return_sequences=True, implementation=1)(Dropout1)
    Dropout2 = Dropout(0.5)(LSTM2)
    LSTM3 = LSTM(lstm_units, return_sequences=True, implementation=1)(Dropout2)
    Dropout3 = Dropout(0.5)(LSTM3)
    TemporalProcessmodel = Model(input=singleinput, output=Dropout3)
    # TemporalProcessmodel.summary()


    # Model 2: Main Structure, starting with independent temporal information encoding
    inputs = Input(shape=(180, 30,)) # The input data is 180 timesteps by 30 features (13 angles+13 energies+4 sEMGs)
                                     # The information each body segment included is the angle and energy

    Angle1 = crop(2, 0, 1)(inputs)
    Acc1 = crop(2, 13, 14)(inputs)
    B1 = concatenate([Angle1, Acc1], axis=-1)
    Anglefullout1 = TemporalProcessmodel(B1)
    AngleAttout1 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout1)
    Blast1 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout1)

    Angle2 = crop(2, 1, 2)(inputs)
    Acc2 = crop(2, 14, 15)(inputs)
    B2 = concatenate([Angle2, Acc2], axis=-1)
    Anglefullout2 = TemporalProcessmodel(B2)
    AngleAttout2 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout2)
    Blast2 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout2)

    Angle3 = crop(2, 2, 3)(inputs)
    Acc3 = crop(2, 15, 16)(inputs)
    B3 = concatenate([Angle3, Acc3], axis=-1)
    Anglefullout3 = TemporalProcessmodel(B3)
    AngleAttout3 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout3)
    Blast3 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout3)

    Angle4 = crop(2, 3, 4)(inputs)
    Acc4 = crop(2, 16, 17)(inputs)
    B4 = concatenate([Angle4, Acc4], axis=-1)
    Anglefullout4 = TemporalProcessmodel(B4)
    AngleAttout4 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout4)
    Blast4 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout4)

    Angle5 = crop(2, 4, 5)(inputs)
    Acc5 = crop(2, 17, 18)(inputs)
    B5 = concatenate([Angle5, Acc5], axis=-1)
    Anglefullout5 = TemporalProcessmodel(B5)
    AngleAttout5 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout5)
    Blast5 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout5)

    Angle6 = crop(2, 5, 6)(inputs)
    Acc6 = crop(2, 18, 19)(inputs)
    B6 = concatenate([Angle6, Acc6], axis=-1)
    Anglefullout6 = TemporalProcessmodel(B6)
    AngleAttout6 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout6)
    Blast6 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout6)

    Angle7 = crop(2, 6, 7)(inputs)
    Acc7 = crop(2, 19, 20)(inputs)
    B7 = concatenate([Angle7, Acc7], axis=-1)
    Anglefullout7 = TemporalProcessmodel(B7)
    AngleAttout7 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout7)
    Blast7 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout7)

    Angle8 = crop(2, 7, 8)(inputs)
    Acc8 = crop(2, 20, 21)(inputs)
    B8 = concatenate([Angle8, Acc8], axis=-1)
    Anglefullout8 = TemporalProcessmodel(B8)
    AngleAttout8 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout8)
    Blast8 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout8)

    Angle9 = crop(2, 8, 9)(inputs)
    Acc9 = crop(2, 21, 22)(inputs)
    B9 = concatenate([Angle9, Acc9], axis=-1)
    Anglefullout9 = TemporalProcessmodel(B9)
    AngleAttout9 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout9)
    Blast9 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout9)

    Angle10 = crop(2, 9, 10)(inputs)
    Acc10 = crop(2, 22, 23)(inputs)
    B10 = concatenate([Angle10, Acc10], axis=-1)
    Anglefullout10 = TemporalProcessmodel(B10)
    AngleAttout10 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout10)
    Blast10 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout10)

    Angle11 = crop(2, 10, 11)(inputs)
    Acc11 = crop(2, 23, 24)(inputs)
    B11 = concatenate([Angle11, Acc11], axis=-1)
    Anglefullout11 = TemporalProcessmodel(B11)
    AngleAttout11 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout11)
    Blast11 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout11)

    Angle12 = crop(2, 11, 12)(inputs)
    Acc12 = crop(2, 24, 25)(inputs)
    B12 = concatenate([Angle12, Acc12], axis=-1)
    Anglefullout12 = TemporalProcessmodel(B12)
    AngleAttout12 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout12)
    Blast12 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout12)

    Angle13 = crop(2, 12, 13)(inputs)
    Acc13 = crop(2, 25, 26)(inputs)
    B13 = concatenate([Angle13, Acc13], axis=-1)
    Anglefullout13 = TemporalProcessmodel(B13)
    AngleAttout13 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(Anglefullout13)
    Blast13 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout13)


    # Feature Concat for Bodily Attention Learning
    # The size of the output from each body segment is k X 1, while k is the number of LSTM hidden units
    # During prior experiments, we found that it is better to keep the dimension k instead of merging them into one

    DATA = concatenate([Blast1, Blast2, Blast3, Blast4, Blast5, Blast6, Blast7, Blast8,
                        Blast9, Blast10, Blast11, Blast12, Blast13
                        ], axis=2)
    # Handy and sufficient Bodily Attention Module
    a = Dense(BodyNum, activation='tanh')(DATA)
    a = Dense(BodyNum, activation='softmax', name='bodyattention')(a)
    attentionresult = multiply([DATA, a])
    attentionresult = Flatten()(attentionresult)
    output = Dense(2, activation='softmax')(attentionresult)

    model = Model(input=inputs, output=output)
    # model.summary()

    return model


    # Main Implementation Part
if __name__ == '__main__':

    list = np.arange(1, 31, 1)  # Number of subjects, can be adjusted to your envrionment
    typelist = np.arange(1, 6, 1) # Number of movement types, can be adjusted to your envrionment
    movement = ['Bend', 'Olg', 'Sits', 'Stsi', 'Rf']

    for index in range(len(list)):
        person = str(list[index])

        if list[index]<13:
            X_train0, X_valid0, y_train, y_valid = loadata('C', person, movement[1]) #In my case, the healthy and CLBP subjects come with different names
        else:
            X_train0, X_valid0, y_train, y_valid = loadata('P', person, movement[1])

        _, samplenum1, dim1 = y_train.shape # Starting from some versions of Keras,
        _, samplenum2, dim2 = y_valid.shape # the first dimension of the label, which is usually '1',
        y_train = np.reshape(y_train, (samplenum1, dim1)) # will not be automatically deleted, so we need to do it manually.
        y_valid = np.reshape(y_valid, (samplenum2, dim2))

        # callback 1: Save the better result after each epoch,
        checkpointer = keras.callbacks.ModelCheckpoint(filepath='PATH+FileName' + person + '.hdf5',
                                                       monitor='val_binary_accuracy', verbose=1,
                                                       save_best_only=True)
        # callback 2: Stop if Acc=1
        class EarlyStoppingByValAcc(keras.callbacks.Callback):
            def __init__(self, monitor='val_acc', value=1.00000, verbose=0):
                super(keras.callbacks.Callback, self).__init__()
                self.monitor = monitor
                self.value = value
                self.verbose = verbose
            def on_epoch_end(self, epoch, logs={}):
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
                if current == self.value:
                    if self.verbose > 0:
                        print("Epoch %05d: early stopping THR" % epoch)
                        self.model.stop_training = True

        callbacks = [
            EarlyStoppingByValAcc(monitor='val_binary_accuracy', value=1.00000, verbose=1),
            checkpointer
                    ]

        model = build_model()
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=ada,
                      metrics=['binary_accuracy'])
        ada = keras.optimizers.Adam(lr=0.003)
        H = model.fit(X_train0, y_train,
                      batch_size=40,
                      epochs=80,
                      shuffle=False,
                      callbacks=callbacks,
                      validation_data=(X_valid0, y_valid))

        print('---This is result for %s th subject---' % person)
        model.load_weights('PATH+FileName' + person + '.hdf5')
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import f1_score
        y_pred = np.argmax(model.predict(X_valid0, batch_size=15), axis=1)
        y_true = np.argmax(y_valid, axis=1)
        cf_matrix = confusion_matrix(y_true, y_pred)
        print(cf_matrix)
        class_wise_f1 = np.round(f1_score(y_true, y_pred, average=None) * 100) * 0.01
        print('the mean-f1 score: {:.2f}'.format(np.mean(class_wise_f1)))
