# BodyAttentionNetwork, Submitted to ACII 2019
This is for the paper 'Learning Bodily and Temporal Attention in Protective Movement Behaviour Detection'


# Preparation
If you want to run the network on EmoPain dataset (http://www.emo-pain.ac.uk/), you need to extract the angle and energy data from it.
The EmoPain dataset will be soon released for a workshop at ACII 2019.

If you want to run the network on your dataset, please make sure you understand how the BANet works.
To do this, please read the paper: 

'Learning Bodily and Temporal Attention in Protective Movement Behaviour Detection.' (2019).

Then you can

i)  change the number of body parts/sensors to your need;

ii) tune the hyperparameter;

iii)use the AttentionScoreExtract.py + TemporalAttenHeatMap.py to analyze your result.


# Code Description
BANet.py is the proposed one with two attention mechanisms, namely bodily-attention and temporal-attention.

BANet-body.py is the variant of BANet only with bodily-attention.

BANet-time.py is the variant of BANet only with temporal-attention.

AttentionScoreExtract.py is used to load the model and obtain the output from specified layers.

TemporalAttenHeatMap.py is to create the heat map for your obtained temporal attention scores.

Within each code, directions are also provided.



# Citation
Find it useful for your project?
Please do remember to cite the papers:

"Learning Bodily and Temporal Attention in Protective Movement Behaviour Detection." (2019). 

"Automatic Detection of Protective Behavior in Chronic Pain Physical Rehabilitation: A Recurrent Neural Network Approach." arXiv preprint arXiv:1902.08990 (2019). https://arxiv.org/abs/1902.08990

