# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:26:09 2023

@author: Tomas

parameter fitting code based on that from Aenne Brielmann:
    https://github.com/aenneb/test-aesthetic-value-model/blob/main/fitPilot.py
    https://github.com/aenneb/test-aesthetic-value-model/blob/main/analysis/d_fit_custom_model_ratings_cv.py

finds values of k (P_switch) and Metrical contrast (spread) (along with scaling parameter m and b) 
that minimize RMSE between delta suprisal and ratings per participant

also generates per-participant predicted ratings for 9 rhythms used in original simulation  
"""



import numpy as np
#import math
#from itertools import product
import pandas as pd
import random
from scipy.optimize import minimize
from deltaSurp_s import deltaSurp #
from getStims import getStims

#%%-------------- 
# set directories;  load rating data or delta Surps for parameter recovery; load stims
#----------

home_dir = '...\\'


recov = False

if recov: #if doing parameter recovery 
    
    df1 = pd.read_csv(home_dir + '...') #load sim ratings
    participantList = list(df1.pID.unique())
else: #otherwise load actual ratings
    
    df = pd.read_csv(home_dir + '...\\df_ratings.csv')
    
    df1 = df.dropna(subset = 'move_ratings')#drop rows of missing ratings, if any
    
    participantList = list(df1.ID.unique())



#get stims
stims = pd.read_pickle(home_dir + '...\\rhythmInfo_60.pkl')


#%%-----------------
# set fixed parameter; set bounds for optimization
#------------

#initialize k and spread to 'max' values
#k = .01
#spread = 1

granularity = 32


# all parameters 
allParams = ['k', 'spread', 'm', 'b']

fixParam = []


bounds = []

# If there are fixParams, exclude them from bounds
if fixParam:
    remaining_params = [param for param in allParams if param not in fixParam]
    
    # Set bounds based on remaining parameters
    for param in remaining_params:
        if param == 'k':
            bounds.append((.01, 1))  # bounds for 'k'
        elif param == 'spread':
            bounds.append((1,4))  #  bounds for 'spread'
        elif param == 'm':
            bounds.append((None, None))  # bounds for 'm'
        elif param == 'b':
            bounds.append((None, None))  # bounds for 'b'
else:
    # If no parameters are fixed, set default bounds for all parameters
    bounds = [(.01, 1), (1, 4), (None, None), (None, None)]  # k, spread, m, b




bounds = tuple(bounds)

nParams = len(bounds) 

#%% --------------------
# unpack parameters (if not fitting all params)
#------------

def unpackParams(parameters, fixParam):
    """
    Read in a list of parameter values and fixParams depending on which are fixed
    then pass parameter values to deltaSurp

    Parameters
    ----------
    parameters : list
        Sorted list of parameter values.

    fixParams: list
        list of parameters to be fixed

    Returns
    -------
    k : float
        value for k
    spread : float
        value for spread
     """
    paramsUsed = 4
     
    if fixParam:
        if 'k' in fixParam:
            
            k = 0.11262987 #from all at once raw, with m and b fixed
            paramsUsed = paramsUsed - 1
        else:
            k = parameters[0]
        if 'spread' in fixParam:
            
            spread = 1.21699844 #from all at once raw
            paramsUsed = paramsUsed - 1
        else:
            spread = parameters[paramsUsed-3]
        if 'm' in fixParam:
            m = 37.98172 # median from init scaling fit on previous ratings
           
            paramsUsed = paramsUsed - 1
        else:
            m = parameters[paramsUsed-2]
        if 'b' in fixParam:
            b = 2.617817 #median from init scaling fit on previous ratings
          
            paramsUsed = paramsUsed - 1
        else:
            b = parameters[paramsUsed-1]
    else:
        k = parameters[0]
        spread = parameters[1]
        m = parameters[2]
        b = parameters[3]
    
    

    return k, spread, m, b

#%% -------------------------
# cost function
#---------------------------

def cost_fn(parameters,  data, rhythms, recov = False, allAtOnce = False):
    """
    Defines the cost function for minimization by returning the RMSE between
    observed and predicted ratings.

    Parameters
    ----------
    k, spread: meter parameters to be fit
    deltaSurp: predicted ratings
    data : pd DataFrame
        Data frame that contains urge to move ratings, syncopation values, age, years of musical training, etc.
    rhythms : str
        identifier string for the rhythmic patterns

    Returns
    -------
    cost : int
        RMSE between predicted and observed ratings.
    """
     #get predicted ratings (deltaMeanSurp_z) for all rhythms that participant rated
    pred_ratings = np.zeros(len(rhythms), dtype= float)

    costs = np.zeros(len(rhythms), dtype = float)

    #unpack parameters
    (k,spread, m, b) = unpackParams(parameters, fixParam)

    if recov: # if doing parameter recovery
        ratings = data['simMoveScore'].values
    else:
        ratings = data['move_ratings'].values #all ratings for that participant


    if allAtOnce: #if fitting to all ratings at once (not per-participant)

        # get full list of patterns corresponding to each rhythm that was rated
        patterns  = [stims[0][rhythm] for rhythm in rhythms] 
              
        
        #get predicted ratings (deltaSurpMean_z) for each pattern
        predList = [deltaSurp(k, spread, m, b, pattern)[0] for pattern in patterns]

        data['pred_ratings'] = predList

        #calculate RMSE between predicted and actual ratings for all rhythms
        cost = np.sqrt(np.mean((data['move_ratings'] - data['pred_ratings'])**2))

    else: #if fitting per participant

        #calculate predicted ratings with deltaSurp, then calculate squared difference from actual ratings
        for i, rhythm in enumerate(rhythms):
            #rhythmIndex = np.where(stims[2] == rhythm)[0][0] #get index of rhythm number that participant rated within stims[2] 
            pattern = stims.loc[stims['name'] == rhythm, 'pattern'].to_numpy()
            pattern = np.tile(pattern[0],4)
            pred_ratings[i]= deltaSurp(k, spread, m , b, granularity, pattern)[1]
            costs[i] = (ratings[i] - pred_ratings[i])**2

        #get RMSE between predicted and actual ratings for all rhythms (single value per participant)
        cost = np.sqrt(np.mean(costs))

    return cost


#%% ------------------ 
# parameter fitting
#----------

def paramFit(data, rhythms, nIter):
    
    """
    uses scipy minimize to find parameter values that lead to best fit between deltaSurp values and participant ratings
    observed and predicted ratings (or preloaded deltaSurps if recov = True).

    Parameters
    ----------
    data: pd dataframe
        dataframe with per rhythm, per participant urge to move ratings
        along with stim info (rhythm complexity) and participant info (age, musical training, PD)

    rhythms : np array
        the rhythms for which to compare ratings and delta surps

    nIter : int
        number of iterations of the fitting

    Returns
    -------
    resDict : dictionary
        dictionary with results of fitting, including RMSE, k, and spread values from best fit (across iterations)
    """
    resList = []
    rhythmsList = []
 

    tmp_res = []
    tmp_rmse = []

    for iteration in range(nIter):

        #np.random.seed(iteration)
        randStartValues = np.random.rand(nParams)

        #scale starting values of free parameters
        if fixParam and 'k' not in fixParam and 'spread' in fixParam:
            randStartValues[0] = randStartValues[0]/10 #scale k starting value
        if fixParam and 'spread' not in fixParam and 'k' not in fixParam: # if fitting k and spread
            randStartValues[0] = randStartValues[0]/10 #scale k starting value
            randStartValues[1] = round(randStartValues[1]+1,2) # scale spread starting value
        if fixParam and 'spread' not in fixParam and 'k' in fixParam:
            randStartValues[0] = round(randStartValues[0]+1,2) # scale spread starting value

        #run the optimization
        thisRes = minimize(cost_fn, randStartValues,
                           args=(  data, rhythms ),
                           bounds = bounds,
                           method = 'SLSQP',
                           options={'maxiter': 1e4, 'ftol': 1e-06})

        print(thisRes.x)
        tmp_res.append(thisRes)
        tmp_rmse.append(thisRes.fun)
       

    #get the results from the seed/run with lowest rmse
    res = tmp_res[tmp_rmse.index(np.nanmin(tmp_rmse))]
    print('parameters for best fit: ' + str(res.x))
    print('best rmse: ' + str(res.fun))
    resList.append(res)
    rhythmsList.append(rhythms)
     

    resDict['rhythms'] += [rhythms]
    
    resDict['res'] += [res]
    resDict['rmse_fit'] += [res.fun]
    
    return resDict

#%% -------------
# optimization; looping through participants or fit all participants at once
# ---------------

nIter = 5

allAtOnce = False

if allAtOnce: #if fitting all participants at once
    resDict = {'res': [], 'rmse_fit': [], 'rmse_pred': [], 'rhythms': []}

    data = df1
         
    rhythms =  data['rhythm'].values
    resDictAll = paramFit(data, rhythms, nIter)

else:
    resDict = {'res': [], 'rmse_fit': [], 'rmse_pred': [], 'participant': [], 'rhythms': []}

    for p in participantList:
        data = df1[df1.ID == p] # all data for that participant
        #data = dfrecov[dfrecov.pID == p]
        rhythms = list(data['stims'].values)
        
        resDictPart  =   paramFit(data, rhythms, nIter)
        #resDictPart, paramDictPart  =   paramFit(data, rhythms, nIter)
        resDictPart['participant'] += [p]
        
        
        print(p)


# to do
# make recov work from here



# %% ---------------------------------------------------------
# Sve the overall res dict to pandas dataframe and save as .csv
# ------------------------------------------------------------

#resDf = pd.DataFrame(resDict)
resDf = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in resDict.items() ]))

resDf["k"] = " "
resDf["spread"] = " "
resDf["m"] = " "
resDf["b"] = " "

for i in range(len(resDf)):
    resDf['k'][i] = resDf.res.values[i].x[0]
    resDf['spread'][i] = resDf.res.values[i].x[1]
    resDf['m'][i] = resDf.res.values[i].x[2]
    resDf['b'][i] = resDf.res.values[i].x[3]

resDf.to_csv('...\\fit_results.csv')


#%% -----------------
# generate per-Participant simMoveScores for orig 9 rhythms 
# --------------------

params  = pd.read_csv('...\\fit_results.csv')

stims = getStims.getStims(stimSet = 'chords1', reps = 4)

dictList = []

granularity = 32

#loop through rows and rhythms, generating a list of delta surps for each pair of param values

for row in params.index:

    k = params.k[row]
    spread = params.spread[row]
    m = params.m[row]
    b = params.b[row]

    for i, rhythm  in enumerate(stims[2]):
        
        pattern = stims[0][i]
        pDict = deltaSurp(k, spread,m, b, granularity, pattern)[2]

        entry = {'ID':params.participant[row], 'rhythm': rhythm, 'syncIndex': stims[1][i]}
        
        pDict.update(entry)

        dictList.append(pDict)        

dfsurps = pd.DataFrame(dictList)

dfsurps.to_csv('...\\predRatings_orig.csv')



