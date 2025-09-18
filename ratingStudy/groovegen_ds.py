# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 15:20:30 2025

@author: Tomas

uses the custom scripts based on the groove generator package to search for rhythmic patterns given delta surprisal values 
or syncopation indices, within constraints

generates are large number, which is then subsampled to get 30 per measure, covering their full range
"""


import numpy as np
import pandas as pd
import sys
import re
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '...\\groovegenerator')

import groovegenerator_t as ggt
import syncIndex
from deltaSurp_s import deltaSurp

outdir =  '...\\'

#DS
#target_list = np.arange(-.045, .085, .0044)

#target_list = np.arange(-0.05, 0.1, .001)

#SI
target_list = np.arange(0, 19, .12)



#%% groove gen ds
def grooveGen(measure, target_list, granularity):
    """
    This function calls searchPattern_ds and looks for patterns based on the measure (delta suprisal or sync Index) and the target list
    It saves out wav and .csv file for each patterns, and csv of info for all patterns

    Parameters
    ----------
    
    measure : str
        'DS' or 'SI'
    target_list : list
        list of DS or SI values 
    
    Returns
    -------
    

    """
    dsList = []
    nList = []
    syncList = []
    nameList = []
    alphaList = []
    
    #params for deltaSurp
    k = .01
    spread = 1
    m = 37.98172 #from initial scaling on young non-musicians, raw (1-5) ratings
    b = 2.617817
    
    for i, target in enumerate(target_list):
                        
        print('looking for a pattern with ' + measure + ' = ' + str(target))
        
        p, out, alpha, success = ggt.searchPattern_ds(measure = measure, target=target, granularity = granularity, timeout=1000, minSnare=5, maxSnare=5, verbose=True) 
        
        if granularity == 32:
            syncInd = syncIndex.SI
            
        elif granularity == 16:
            syncInd = syncIndex.SI16
            
        else:
            print('Incorrect input for granularity')
        
        if p is not None: #if found a rhythm
             
             onsets = p[1]
             nOnsets = sum(onsets)
             nList.append(nOnsets)
             #print(f'nSnares = {nSnares}')
             
             if measure == 'DS':    
                 ds = out
                 SI = syncInd(p[1])[0]
                 outWav = 'WAVs_DS\\'
                 outCSV = 'CSVs_DS\\'
                 outName = 'DS'
                 nRound = 3
             elif measure == 'SI':
                 SI = out
                 ds = deltaSurp(k, spread, m, b, granularity, p[1])[0]
                 outWav = 'WAVs_SI\\'
                 outCSV = 'CSVs_SI\\'
                 outName = 'SI'
                 nRound = 0
             else:
                 print('Incorrect input for measure')
             
             dsList.append(ds)
             syncList.append(SI)
             alphaList.append(alpha)
            # save stuff out
             stimName = outName + '_' + str(i+1) + '_' + str(round(out,nRound))
             nameList.append(stimName)
             
             ggt.savePattern(p, outdir + outCSV + stimName )
             ggt.generate_wav(p, loops = 4, saveName = outdir + outWav + stimName, tempo = 192, customSound = 'chords')    
            #ggt.generate_midi(p[0], loops = 2, saveName = mididir + 'ggDS_' + str(i))
             print(f'found a pattern with delta Surp  = {ds}, nOnsets = {nOnsets}, and SI = {SI}')
                
                
        else:  
            # move on to next
             print('searchPattern failed for '+ measure + ' = ' + str(target) )
        
    
    
    dsDict = { 'deltaSurp': dsList, 'nOnsets': nList, 'syncIndex': syncList, 'name': nameList, 'alpha':alphaList}    
    
    dfDict = pd.DataFrame(dsDict)
    
    dfDict.to_csv('...' + outName +'_rhythmInfo.csv', index = False)



#%%----------
# get subset and copy to new folder
# ----

import shutil

def copySubset(measure = 'SI', nStim = 30):
    """
    copies subset of wav and csv files in consecutive order, based on number of stims in subset 

    assumes specific folder structure
    """    
    homeDir = '...\\'
    
    #load rhythm info csv
    info = pd.read_csv(homeDir + measure + '_rhythmInfo.csv')
    if measure == 'DS':
        info = info.sort_values(by = ['deltaSurp'])
    elif measure == 'SI':
        info = info.sort_values(by = ['syncIndex'])
    
    wavDirect = homeDir + 'WAVs_' + measure + '\\'
    csvDirect = homeDir + 'CSVs_' + measure + '\\'
    
    wavDest = homeDir + 'WAVs_' + measure + '_30\\'
    csvDest = homeDir + 'CSVs_' + measure + '_30\\'
    
    nFiles = len(info)
    # get names
    # for names 1, 5,10, 15, etc., 
    #index = np.arange(0, nFiles , round(nFiles/nStim)).tolist() 
    index = np.linspace(0, nFiles - 1, nStim, dtype=int).tolist()
    #add .wav, .csv, copy from WAV and CSV folder to 30 folder
    names = info['name'][index]
    for i, name in enumerate(names):
        wavFile = wavDirect + name + '.wav'
        csvFile = csvDirect + name + '.csv'
        shutil.copy(wavFile, wavDest)
        shutil.copy(csvFile, csvDest)
        
    info_30 = info.iloc[index]
    
    info_30.to_csv('...\\' + measure +'_rhythmInfo_30.csv', index = False)


#%% -------------
# load DS_30 and SI_30 csvs and generate wavs with different medium chords
#---------------

measure = 'SI'

homeDir =  '...\\'

csvDir = homeDir + 'CSVs_' + measure + '_30\\'
outDir = homeDir + 'WAVs_' + measure + '_30\\'

#all real rhythms
fnames = np.array(os.listdir(csvDir))
np.random.shuffle(fnames)

for i in range(20, 30):
    p = ggt.loadPattern(csvDir + fnames[i])
    ggt.generate_wav(p, loops = 4, saveName = outDir + re.sub('.csv', '',fnames[i]), tempo = 192, customSound = 'chords')  
    
    
 

    
