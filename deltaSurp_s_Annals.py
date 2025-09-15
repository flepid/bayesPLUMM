# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 19:05:11 2022

@author: Tomas Matthews, Jon Cannon


"""

import numpy as np
import math
from statistics import mean


def deltaSurp(k, spread, m, b, granularity, pattern):
    """
    takes a single stream rhythm and calculates posterior and surprisal
    based on metric and uniform weights (i.e., template)
    metric weights based on LH-L, Lerhdahl & Jackendoff
      
    Parameters
    ----------
    k : float
        weights prior of each time step
        probability of switching templates
        inverse is meter persistence
             
    spread : float
        scales metric weights
        degree of difference between high or low probabilities 
        higher values indicate more flattened template probabilities
        inverse is metrical contrast
        
    m : float
        scaling weight for delta Surp
        
    b : float
        intercept for scaling delta surp
    
    pattern : np array
        array of ones and zeros delineating rhythm onsets
    
    Returns
    -------
    deltaSurp: float
        difference in surprisal (averaged within rhythm) between rhythm with and without metronome
    
    simMoveScore: float
        delta surprisal scaled to match ratings by fitting m and b params
        
    paramDict : dictionary
        dictionary with all the per timestep parameter info including posteriors, suprisal, metric weights etc.
        useful for plotting but not used in parameter fitting
      
    """
    
    rhythm_reps = int(len(pattern)/32)
    nBeats = len(pattern)
    
    t_list = np.arange(nBeats) # time steps
    nTemp = 2 #number of templates
    
        
    #initialize outputs
    joint = np.zeros([nTemp, nBeats], dtype = float) #joint probabilities
    post = np.zeros([2, nTemp, nBeats+1], dtype = float) # posterior probabilities
    surprisal = np.zeros( [2, nBeats], dtype = float)
    meanSurp = np.zeros(2, dtype = float)
    spreadCol =   list()
    resCol = list()
    
    
    ## changing granularity   
    # indices for changing granularity 
    # 32nd note indices
    i32 = np.arange(1, 32, 2).tolist()
    #i32 = np.arange(1, 32, 2)
    
    #16th note indices
    i16 = np.arange(2, 32, 4).tolist()
           
    #metric weights     
    maxWeights = np.array([.9,.1,.3,.1,.5,.1,.3,.1,.7,.1,.3,.1,.5,.1, .3,.1,.8,.1,.3,.1,.5,.1,.3,.1,.7,.1,.3,.1,.5,.1,.3,.1 ])
    
    MQ = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) # quarter note
    #MQ = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]) #eighth note
    #MQ = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  #half note
    
    #applying this to weights instead of max weights, so that spread does not affect e.g., the 32n note positions 
    if granularity == 32:
        # 32nd note granularity 
        maxWeights = maxWeights    
    
    elif granularity == 16:        
           
        # replace .1's with .5
        #weights = np.array(weights)
        #maxWeights[i32] = .5
        maxWeights = np.delete(maxWeights, i32)
        MQ = np.delete(MQ, i32)

    elif granularity == 8:    
           
        #actually changes granularity to quarter notes since eighth notes already .5's
        maxWeights[i32] = .5
        maxWeights[i16] = .5
    else:
        print('Incorrect input for granularity')
           
    resCol.append(granularity)
   
    
    # apply spread to maxWeights
    #weights = [(x - sum(maxWeights)/len(maxWeights)) / spread+ sum(maxWeights)/len(maxWeights) for x in maxWeights]
    weights =  mean(maxWeights) + (maxWeights - mean(maxWeights)) * spread 
    
    spreadCol.append(spread)
    
     
    
    #repeat
    weights = np.tile(weights, rhythm_reps)
    MQ = np.tile(MQ, rhythm_reps)
    
    #uniform weights (null)
    uniWeights = np.tile(mean(weights), len(weights))
    #uniWeights = np.tile(.55, len(weights))      
      
    
    # uniform and metric weights 
    lambda_ = np.zeros([nTemp, nBeats], dtype = float) #priors
    lambda_[0,] = uniWeights # uniform prior
    #lambda_[0,] = weights
    
    # uniform prior
    lambda_[1,]= weights #  metric prior
    
    #initial prior as uniform
    post[:,:,0] = 1/nTemp
    
               
    #weights on posterior
    Pw = np.zeros([nTemp, nBeats], dtype = float)  
    
    mPattern = pattern + MQ
    mPattern[mPattern ==2] = 1
    
    patterns = [pattern, mPattern]   
    
    
    for r in range(len(patterns)):
        for i in t_list:
            post[r, :, i+1] = post[r, :, i]
            
            
            if patterns[r][i] == 1: #if there is an event
                P = lambda_[:, i]
            else:
                P = 1-lambda_[:, i]
            # joint prior of current time step 
            # 
            
            # generate a weight for next each step
            #  using posterior from the previous timestep
            # plus the difference between posteriors across the two templates, weighted by k
                                            
            Pw[0,i] = post[r, 0, i+1]+ k*(post[r, 1, i+1] - post[r, 0, i+1]) #weighting posterior from the previous beat at each step
            Pw[1,i] = post[r, 1, i+1]+ k*(post[r, 0, i+1] - post[r, 1, i+1])# i.e., so  the prior includes the weighted posterior from the previous timestep
                        
            
            joint[:, i] = P[:]*Pw[:,i]
                       
            denom = np.sum(joint[:,i]) # 

            
            post[r, :, i+1] = joint[:,i]/denom 
        
            #calculate surprisal as negative log of p(event)     
            #try:
            surprisal[r, i] = -math.log(denom)
            #except:
                #import pdb
                #pdb.set_trace()
        meanSurp[r] = mean(surprisal[r])
            
    deltaSurp = meanSurp[0] - meanSurp[1] #w/out metronome - with metronome
         
    #scale with fit scaling parameters
    simMoveScore = m*deltaSurp + b
    
    
    meanPostnMetroUni = mean(post[0,0,:])  
    meanPostnMetroMet = mean(post[0,1,:])
    meanPostMetroUni = mean(post[1,0,:])  
    meanPostMetroMet = mean(post[1,1,:])
    
    # put parameter values in dictionary
    paramDict = { 'pattern': pattern,  'surprisal_nMetro': surprisal[0,], 'surprisal_Metro': surprisal[1,], 'm': m, 'b': b,
                     'deltaSurp': deltaSurp, 'simMoveScore': simMoveScore, 'meanSurp_nMetro': meanSurp[0], 'meanSurp_Metro': meanSurp[1],
                     'postUni_nMetro': post[0,0,:], 'postMet_nMetro': post[0, 1,:], 'postUni_Metro': post[1,0,:], 'postMet_Metro': post[1, 1,:], 
                     'meanPostUni_nMetro': meanPostnMetroUni, 'meanPostMet_nMetro': meanPostnMetroMet,  'meanPostUni_Metro': meanPostMetroUni, 'meanPostMet_Metro': meanPostMetroMet,
                      'spread': spread, 'granularity': resCol, 'k': k, 'mWeights': weights, 'uniWeights': uniWeights}
   
    
    return deltaSurp, simMoveScore, paramDict

