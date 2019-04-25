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



model = build_model() # You can call the model for BANet or yours
model.compile(loss=keras.losses.binary_crossentropy,
             optimizer='adam',
             metrics=['binary_accuracy'])

model.load_weights('PATH+FileName.hdf5') # Load the pre-trained weight

Timeatten1 = model.get_layer('TemporalAtten1').output # Get the output of layer by its namea,
Timeatten2 = model.get_layer('TemporalAtten2').output # which need to be specified in your model layers with argument 'name'
Timeatten3 = model.get_layer('TemporalAtten3').output
Timeatten4 = model.get_layer('TemporalAtten4').output
Timeatten5 = model.get_layer('TemporalAtten5').output
Timeatten6 = model.get_layer('TemporalAtten6').output
Timeatten7 = model.get_layer('TemporalAtten7').output
Timeatten8 = model.get_layer('TemporalAtten8').output
Timeatten9 = model.get_layer('TemporalAtten9').output
Timeatten10 = model.get_layer('TemporalAtten10').output
Timeatten11 = model.get_layer('TemporalAtten11').output
Timeatten12 = model.get_layer('TemporalAtten12').output
Timeatten13 = model.get_layer('TemporalAtten13').output
Timeatten = concatenate([Timeatten1, Timeatten2, Timeatten3, Timeatten4, Timeatten5, Timeatten6,
                     Timeatten7, Timeatten8,Timeatten9, Timeatten10, Timeatten11, Timeatten12,
                     Timeatten13],axis=-1)

visualizemodel = Model(inputs=model.input, outputs=Timeatten) # Create a new model that will take in the same input but
                                                              # give the layer output of another model
layeroutput = visualizemodel.predict(X_valid, batch_size=40)
scipy.io.savemat('PATH+FileName.mat',
             {'name': layeroutput})    #Data will be save in the mat file