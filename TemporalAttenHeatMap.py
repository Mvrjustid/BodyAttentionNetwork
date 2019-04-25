# Plot the heatmap of temporal attention
import numpy as np
import scipy.io
import h5py
import os
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

#-----------------------------------------||||||||||||||||||||||||||||||||||||||||||||||||||

# Reference:

# "Learning Bodily and Temporal Attention in Protective Movement Behavior Detection"
#  arxiv preprint arxiv:1904.10824 (2019)

# "Automatic Detection of Protective Behavior in Chronic Pain Physical Rehabilitation: A Recurrent Neural Network Approach."
#  arXiv preprint arXiv:1902.08990 (2019).

# If you find the code useful, please cite the paper above
# : )

#-----------------------------------------||||||||||||||||||||||||||||||||||||||||||||||||||


data = scipy.io.loadmat('PATH+FileName.mat') # This should be the temporal attention scores
                                             # with size： Bodysegments/sensor numbers X Timesteps
                                             # and is better normalizaed to 0-1
heatdata=data['variablename']
# ax = sb.heatmap(heatdata,vmin=0.1,vmax=1,cmap='jet')
ax = sb.heatmap(heatdata,vmin=0,vmax=1,cmap=sb.cubehelix_palette(n_colors=100,light=.95,dark=.08))
# Search seaborn cmap to learn
# how to customize the color setting
plt.show()