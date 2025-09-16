# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:39:02 2024

@author: Tomas

load chord/TM rhythms or FR rhythms

updated to include larger chordRhythms2 set
"""

import os
os.chdir('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Scripts')

import numpy as np
from chordRhythm_stimuli import chordRhythms
from chordRhythm_stimuli2 import chordRhythms2
from fitchrosenfeld_stimuli import FR2007_RHYTHMS

def getStims(stimSet = 'chords1', reps = 4): 
    nSteps = len(chordRhythms[0][0])#assumes all rhythms are same lengthe regardless of set
    nStepsT = nSteps*reps
    if stimSet == 'chords1':
        stims = chordRhythms
        syncList = []
        rhythmNames = []
        rhythmIndex = np.zeros(len(stims), dtype = int)
        rhythmList= [np.zeros([nStepsT,], dtype = int) for _ in stims]
        for i, f in enumerate(stims):
            rhythmList[i][0:nStepsT] = np.tile(f[0],reps)
            sync = f[1]
            rhythmIndex[i]= f[2]
            rhythmName = f[3]
            syncList.append(sync)
            rhythmNames.append(rhythmName)
    elif stimSet == 'chords2':
        stims = chordRhythms2
        syncList = []
        rhythmNames = []
        rhythmIndex = np.zeros(len(stims), dtype = int)
        rhythmList= [np.zeros([nStepsT,], dtype = int) for _ in stims]
        for i, f in enumerate(stims):
            rhythmList[i][0:nStepsT] = np.tile(f[0],reps)
            sync = f[1]
            rhythmIndex[i]= f[2]
            rhythmName = f[3]
            syncList.append(sync)
            rhythmNames.append(rhythmName)
    elif stimSet == 'FR': 
        
        syncList = []
        rhythmIndex = np.zeros(len(FR2007_RHYTHMS), dtype = int)
        rhythmList= [np.zeros([nStepsT,], dtype = int) for _ in FR2007_RHYTHMS]
        for i, f in enumerate(FR2007_RHYTHMS):
            r16 = f[0]
            sync = f[1]
            rhythmIndex[i]= f[2]
            rhythmList[i][0:nStepsT:2] = np.tile(r16,reps)
            syncList.append(sync)
    return  rhythmList, syncList, rhythmIndex, rhythmNames