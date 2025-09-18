# bayesPLUMM
Code for project investigating the urge to move to music as active inference.
Includes code for analyses and plots in manuscript under review at the Annals of the New York Academy of Sciences entitled 'An active inference model of meter perception and the urge to move to music', Tomas E. Matthews, Peter Vuust, & Jonathan Cannon


The **bayesPLUMM_model** folder contains code for the model (bayesPLUMM_deltaSurp.py), the rhythmic patterns (bayesPLUMM_chordRhythm_stimuli.py), accessing the rhythmic patterns (bayesPLUMM_getStims.py), and making the plots in the manuscript (bayes_PLUMM_plots.py; main one which calls the others).

Note: model parameters have different names in the code and the manuscript: spread = metrical contrast (C); k = P_switch

The **ratingStudy** folder contains code for generating the rhythmic patterns (groovegen_ds.py), for analyzing the ratings (GG_DS_1_ratings.Rmd), for fitting the switch and contrast parameters to the ratings (GG_DS_paramFit.py) and for plotting the resulting fits (GG_DS_1.1.Rmd).

Note: groovegen_ds.py relies heavily on customized versions of functions from the groovegenerator package. The original package is found here https://github.com/OleAd/GrooveGenDist. I will upload the custom versions of those scripts ASAP. 


Data files can be found here: 10.17605/OSF.IO/QJN6Y
