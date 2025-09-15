# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:49:41 2024

@author: Tomas
"""
import os
os.chdir('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Scripts')

import numpy as np
import pandas as pd
from itertools import product 
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean

from deltaSurp_s import deltaSurp
import getStims




#load stims
stims = getStims.getStims(stimSet = 'chords1', reps = 1)

patterns  = stims[0]


ms = [37.98172] #from initial scaling on young non-musicians, raw (1-5) ratings
bs = [2.617817]

#set parameter values 
#ks = [.01]
#spreads = [1]
ks = [.01,.1,.2,.3]
spreads = [0.1,.33,.66, 1]

granularities = [32]


#calculate suprisal, delta surprisal, posteriors, etc. values 
surpDict = [deltaSurp(k, spread, m, b, granularity,pattern)[2] for (k, spread, m, b, granularity, pattern) in product(ks, spreads, ms, bs, granularities, patterns)]


#add columns with potentially useful labels, e.g.,  rhythm, syncIndex, complexity
dfm = pd.DataFrame(surpDict)
dfm['rhythmName'] = np.tile(stims[3], len(spreads)*len(ks))
#dfm['rhythmName'] = np.repeat(['L3', 'L6', 'L7', 'S', 'R', 'M', 'H1', 'H3', 'H7'],len(spreads),axis = 0)
dfm['complexity'] = np.tile(np.repeat(['Low',  'Medium',  'High'], len(patterns)/3),len(spreads)*len(ks))
dfm['syncIndex'] =  np.tile(stims[1], len(spreads)*len(ks))
dfm['rhythm'] = np.tile(stims[2], len(spreads)*len(ks))


#  rename columns for nice plot friendly names
dfm = dfm.rename(columns={'complexity': 'Syncopation', 
                         'deltaSurp':'Difference raw', 'simMoveScore':'Difference (scaled)',
                          'meanSurp_Metro':'Stimulus + Metronome', 'meanSurp_nMetro':'Stimulus',
                          'deltaMeanSurp':'Difference'})

#melt to long format, for plotting of mean values
dfm_long = pd.melt(dfm, id_vars=['Syncopation', 'syncIndex', 'rhythm', 'spread',
                            'granularity','k', 'pattern'], var_name='Type', value_name='Surprisal', 
              value_vars=['Stimulus', 'Stimulus + Metronome', 'Difference raw', 'Difference (scaled)'])


#%% suprisal, raw delta surprisal plots for orig rhythms



#surprisal, with and without metronome, plus delta surp
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ax1, ax2 = axs
cs = sns.color_palette('Set1', 3)

sns.pointplot(data=dfm_long[~dfm_long['Type'].str.contains('Difference')], x='Syncopation', y='Surprisal', palette='Set1', hue='Type', dodge=True, errorbar = 'ci', capsize=.0, ax=ax1)
ax1.set_ylabel('Mean Surprisal', fontsize=18)
ax1.legend(loc='upper left', fontsize = 12)
ax1.set_xlabel('Rhythmic Complexity', fontsize=18)
ax1.tick_params(left=False, labelsize = 12) 

#ax1.set_ylim(.2, .6)
#ax1.set_xticklabels(('Low', 'Medium', 'High'))

sns.pointplot(data=dfm_long[dfm_long.Type == 'Difference raw'], x='Syncopation', y='Surprisal', color=cs[-1], dodge=True, errorbar ='ci', capsize=.0, ax=ax2)
ax2.set_ylabel(r'$\Delta$ Surprisal', fontsize=18)
ax2.axhline(0, alpha=0.75, c='lightgray', ls='--')
ax2.set_xlabel('Rhythmic Complexity', fontsize=18)
ax2.tick_params(left=False, bottom = False, labelsize = 12) 
#ax2.set_ylim(-.06, .075)
#ax2.set_xticklabels(('Low', 'Medium', 'High'))
#ax2.legend(loc=0)


#fig.suptitle(r'$\lambda_u$ = 0.55',fontsize=24);

fig.tight_layout()

plt.savefig('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Manuscripts\\Annals\\Figures\\Final- Revision 2\\surp_deltaSurp2.png', dpi = 300)
#plt.show()



#%%deltasurp or simMoveScore over various combinations of k and spread



cs = sns.color_palette('Set1', 3)
sns.set(style = 'white', rc={'figure.figsize':(5,4)}) 

for(a,b) in product(ks, spreads):
    plt.figure()
    sns.pointplot(x='Syncopation', y='Difference raw', 
        data=dfm[(dfm.k == a) & (dfm.spread ==b)], color=cs[-1], dodge=True, capsize=.0).set_title( '$P_{switch}$ = ' + str(a) + ', $C$ = ' + str(b), fontdict ={ 'fontsize': 18})
    
    plt.ylim(-0.09,0.08)
    plt.axhline(0, alpha=0.75, c='lightgray', ls='--')
    plt.ylabel(r'$\Delta$ Surprisal', fontsize=18)
    plt.xlabel('Rhythmic Complexity', fontsize=18)
    plt.tight_layout() 
    plt.show()
    #plt.savefig('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Manuscripts\\Annals\\Figures\\Pieces\\spread_k\\deltaSurp'+ str(round(a,2)) + '_' + str(round(b,2)) + '.png', dpi = 300)


#%% mean surprisal  over various combinations of k and spread

dfm_long2 = dfm_long[~dfm_long['Type'].str.contains('Difference')]

cs = sns.color_palette('Set1', 3)
sns.set(style = 'white', rc={'figure.figsize':(5,4)}) 

for(a,b) in product(ks, spreads):
    plt.figure()
    sns.pointplot(x='Syncopation', y='Surprisal',palette='Set1', hue='Type', dodge=True, errorbar = 'ci', capsize=.0,
        data=dfm_long2[(dfm_long2.k == a) & (dfm_long2.spread ==b)]).set_title( '$P_{switch}$ = ' + str(a) + ', $C$ = ' + str(b), fontdict ={ 'fontsize': 18})
    plt.legend(loc='upper left', fontsize = 12)
    plt.ylim(0.25, .6)
    #plt.axhline(0, alpha=0.75, c='lightgray', ls='--')
    plt.ylabel('Mean Surprisal', fontsize=18)
    plt.xlabel('Rhythmic Complexity', fontsize=18)
    plt.tight_layout() 
    #plt.show()
    plt.savefig('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Manuscripts\\Annals\\Figures\\Pieces\\spread_k\\meanSurp'+ str(round(a,2)) + '_' + str(round(b,2)) + '.png', dpi = 300)




#%% set up df for rhythm and raw plots
dfm['pattern2'] = ' '
dfm['pattern2'] = dfm['pattern2'].astype(object)

nStim = len(dfm)



for i in range(nStim):
    dfm['pattern2'][i] = np.where(dfm.pattern[i][0:32] ==1,nStim-i,0) #for rhythm schematics

df_exploded = dfm.explode('pattern2')
df_exploded['pattern2'] = df_exploded['pattern2'].astype(float) 
df_exploded['pattern3'] = df_exploded['pattern2']
df_exploded.pattern3[df_exploded.pattern3 == 0] = -2
df_exploded.pattern3[df_exploded.pattern3 >= 1] = 0


df_exploded['timeStep'] = np.tile(np.arange(32)+1, nStim)


#%% set up posterior and suprisal plots
# only need one rhythm per complexity so subset df




indices = [0,3,8]

df1 = df_exploded[df_exploded.index.isin(indices)]



#arrays for strong and weak beats
qbeat = np.arange(0,129,8) #quarter notes
mbeat = np.arange(0,129,32)# down beat
ebeat =  np.arange(4,129,8) #eighth notes
sbeat = np.arange(2,129,4)

#add metronome
qbeat2 = np.arange(0,32,8)
metronome= np.zeros(32)
metronome[qbeat2] = 1
metronome[metronome == 0] = -2
metronome[metronome > 0] = 0
df1['Metronome'] = np.tile(metronome,len(indices))

#%% posteriors without metronome

nSteps = len(df1.postMet_nMetro[df1.Syncopation == 'Low'].iloc[0])

fig, axs = plt.subplots(3, 1, figsize=(6, 6))
ax1, ax2, ax3 = axs

ax1.plot(np.arange(nSteps), df1.postMet_nMetro[df1.Syncopation == 'Low'].iloc[0])

ax1.set_title('Low ', fontsize = 18)
ax1.set_ylim(-0.1,1.1)
ax1.set_xlim(-2,130)

ax1.vlines(x = qbeat, ymin = -0.1, ymax = 2, ls = '-', lw = 4, colors = 'lightgrey', zorder = 0)
ax1.vlines(x = ebeat, ymin = -0.1, ymax = 2, ls = '-',colors = 'lightgrey', zorder = 0)
ax1.vlines(x = mbeat, ymin = -0.1, ymax = 2, ls = '-',colors = 'grey', zorder = 0)


#add rhythm
ax1.scatter( x = np.arange(nSteps-1),
              y = np.tile(df1.pattern3[df1.Syncopation == 'Low'],4), s = 10,c  = 'red')

ax1.tick_params(axis='both',  pad=.002)

ax2.plot(np.arange(nSteps),df1.postMet_nMetro[df1.Syncopation == 'Medium'].iloc[0])
ax2.set_ylabel('Posterior probability of metered template $P^{post}_{metered} (t)$', fontsize = 16)

ax2.set_title('Medium', fontsize = 18)
ax2.set_ylim(-0.1,1.1)
ax2.set_xlim(-2,130)

ax2.vlines(x = qbeat, ymin = -0.1, ymax = 2, ls = '-', lw = 4, colors = 'lightgrey', zorder = 0)
ax2.vlines(x = ebeat, ymin = -0.1, ymax = 2, ls = '-',colors = 'lightgrey', zorder = 0)
ax2.vlines(x = mbeat, ymin = -0.1, ymax = 2, ls = '-',colors = 'grey', zorder = 0)

ax2.scatter( x = np.arange(nSteps-1),
              y = np.tile(df1.pattern3[df1.Syncopation == 'Medium'],4), s = 10, c = 'red')

ax2.tick_params(axis='both',  pad=.002)

ax3.plot(np.arange(nSteps), df1.postMet_nMetro[df1.Syncopation == 'High'].iloc[0])
ax3.set_title('High', fontsize = 18)
ax3.set_ylim(-0.1,1.1)
ax3.set_xlim(-2,130)

ax3.vlines(x = qbeat, ymin = -0.1, ymax = 2, ls = '-', lw = 4, colors = 'lightgrey', zorder = 0)
ax3.vlines(x = ebeat, ymin = -0.1, ymax = 2, ls = '-',colors = 'lightgrey', zorder = 0)
ax3.vlines(x = mbeat, ymin = -0.1, ymax = 2, ls = '-',colors = 'grey', zorder = 0)

ax3.scatter( x = np.arange(nSteps-1),
              y = np.tile(df1.pattern3[df1.Syncopation == 'High'],4), s = 10,c = 'red')
ax3.set_xlabel('Time Step', fontsize = 16)

ax3.tick_params(axis='both',  pad=.002)

fig.suptitle('Stimulus', fontsize = 24, x = .55)

fig.tight_layout()

plt.savefig('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Manuscript\\Final Figures\\raw_posts_Surps\\post_stim.png', dpi = 300)


#%% posteriors with metronome

nSteps = len(df1.postMet_Metro[df1.Syncopation == 'Low'].iloc[0])

fig, axs = plt.subplots(3, 1, figsize=(6, 6))
ax1, ax2, ax3 = axs



ax1.plot(np.arange(nSteps),df1.postMet_Metro[df1.Syncopation == 'Low'].iloc[0])

ax1.set_title('Low ', fontsize = 18)
ax1.set_ylim(-0.1,1.1)
ax1.set_xlim(-2,130)

ax1.vlines(x = qbeat, ymin = -0.1, ymax = 2, ls = '-', lw = 4, colors = 'lightgrey', zorder = 0)
ax1.vlines(x = ebeat, ymin = -0.1, ymax = 2, ls = '-',colors = 'lightgrey', zorder = 0)
ax1.vlines(x = mbeat, ymin = -0.1, ymax = 2, ls = '-',colors = 'grey', zorder = 0)

ax1.scatter( x = np.arange(nSteps-1),
              y = np.tile(df1.pattern3[df1.Syncopation == 'Low'],4), s = 10,c  = 'red')

ax1.scatter( x = np.arange(nSteps-1),
              y = np.tile(df1.Metronome[df1.Syncopation == 'Low'],4), s = 10,c  = 'green')

ax1.tick_params(axis='both',  pad=.002)


ax2.plot(np.arange(nSteps),df1.postMet_Metro[df1.Syncopation == 'Medium'].iloc[0])
#ax2.set_ylabel('Posterior', fontsize = 18)

ax2.set_title('Medium', fontsize = 18)
ax2.set_ylim(-0.1,1.1)
ax2.set_xlim(-2,130)

ax2.vlines(x = qbeat, ymin = -0.1, ymax = 2, ls = '-', lw = 4, colors = 'lightgrey', zorder = 0)
ax2.vlines(x = ebeat, ymin = -0.1, ymax = 2, ls = '-',colors = 'lightgrey', zorder = 0)
ax2.vlines(x = mbeat, ymin = -0.1, ymax = 2, ls = '-',colors = 'grey', zorder = 0)

ax2.scatter( x = np.arange(nSteps-1),
              y = np.tile(df1.pattern3[df1.Syncopation == 'Medium'],4), s = 10,c  = 'red')

ax2.scatter( x = np.arange(nSteps-1),
              y = np.tile(df1.Metronome[df1.Syncopation == 'Medium'],4), s = 10,c  = 'green')

ax2.tick_params(axis='both',  pad=.002)

ax3.plot(np.arange(nSteps),df1.postMet_Metro[df1.Syncopation == 'High'].iloc[0])
ax3.set_title('High', fontsize = 18)
ax3.set_ylim(-0.1,1.1)
ax3.set_xlim(-2,130)

ax3.vlines(x = qbeat, ymin = -0.1, ymax = 2, ls = '-', lw = 4, colors = 'lightgrey', zorder = 0)
ax3.vlines(x = ebeat, ymin = -0.1, ymax = 2, ls = '-',colors = 'lightgrey', zorder = 0)
ax3.vlines(x = mbeat, ymin = -0.1, ymax = 2, ls = '-',colors = 'grey', zorder = 0)

ax3.scatter( x = np.arange(nSteps-1),
              y = np.tile(df1.pattern3[df1.Syncopation == 'High'],4), s = 10,c  = 'red')

ax3.scatter( x = np.arange(nSteps-1),
              y = np.tile(df1.Metronome[df1.Syncopation == 'High'],4), s = 10,c  = 'green')
ax3.set_xlabel('Time Step', fontsize = 16)

ax3.tick_params(axis='both',  pad=.002)

fig.suptitle('Stimulus + Metronome', fontsize = 24, x = .55)

fig.tight_layout()

#plt.show()
plt.savefig('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Manuscript\\Final Figures\\raw_posts_Surps\\post_stimMet.png', dpi = 300)




#%% surprisal plot without metronome

nSteps = len(df1.surprisal_nMetro[df1.Syncopation == 'Low'].iloc[0])

fig, axs = plt.subplots(3,1, figsize=(6, 6))
ax1, ax2, ax3 = axs

ax1.plot(np.arange(nSteps),df1.surprisal_nMetro[df1.Syncopation == 'Low'].iloc[0])

ax1.set_title('Low ', fontsize = 18)
ax1.set_ylim(-0.2,2.3)
ax1.set_xlim(-2,130)

ax1.vlines(x = qbeat, ymin = -0.2, ymax = 3.2, ls = '-', lw = 4, colors = 'lightgrey', zorder = 0)
ax1.vlines(x = ebeat, ymin = -0.2, ymax = 3.2, ls = '-',colors = 'lightgrey', zorder = 0)
ax1.vlines(x = mbeat, ymin = -0.2, ymax = 3.2, ls = '-',colors = 'grey', zorder = 0)

ax1.scatter( x = np.arange(nSteps),
              y = np.tile(df1.pattern3[df1.Syncopation == 'Low'],4), s = 10,c  = 'red')

ax1.tick_params(axis='both',  pad=.002)

ax2.plot(np.arange(nSteps),df1.surprisal_nMetro[df1.Syncopation == 'Medium'].iloc[0])
ax2.set_ylabel('Surprisal $S(t)$', fontsize = 16)
ax2.set_title('Medium', fontsize = 18)
ax2.set_ylim(-0.2,2.3)
ax2.set_xlim(-2,130)

ax2.vlines(x = qbeat, ymin = -0.2, ymax = 3.2, ls = '-', lw = 4, colors = 'lightgrey', zorder = 0)
ax2.vlines(x = ebeat, ymin = -0.2, ymax = 3.2, ls = '-',colors = 'lightgrey', zorder = 0)
ax2.vlines(x = mbeat, ymin = -0.2, ymax = 3.2, ls = '-',colors = 'grey', zorder = 0)

ax2.scatter( x = np.arange(nSteps),
              y = np.tile(df1.pattern3[df1.Syncopation == 'Medium'],4), s = 10,c  = 'red')

ax2.tick_params(axis='both',  pad=.002)

ax3.plot(np.arange(nSteps),df1.surprisal_nMetro[df1.Syncopation == 'High'].iloc[0])
ax3.set_title('High', fontsize = 18)
ax3.set_ylim(-0.2,2.3)
ax3.set_xlim(-2,130)

ax3.vlines(x = qbeat, ymin = -0.2, ymax =3.2, ls = '-', lw = 4, colors = 'lightgrey', zorder = 0)
ax3.vlines(x = ebeat, ymin = -0.2, ymax = 3.2, ls = '-',colors = 'lightgrey', zorder = 0)
ax3.vlines(x = mbeat, ymin = -0.2, ymax = 3.2, ls = '-',colors = 'grey', zorder = 0)

ax3.scatter( x = np.arange(nSteps),
              y = np.tile(df1.pattern3[df1.Syncopation == 'High'],4), s = 10,c  = 'red')
ax3.set_xlabel('Time Step', fontsize = 16)

ax3.tick_params(axis='both',  pad=.002)

fig.tight_layout()


plt.savefig('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Manuscript\\Final Figures\\raw_posts_Surps\\surp_stim.png', dpi = 300)

#%% surprisal plot with metronome

nSteps = len(df1.surprisal_Metro[df1.Syncopation == 'Low'].iloc[0])

fig, axs = plt.subplots(3, 1, figsize=(6, 6))
ax1, ax2, ax3 = axs

ax1.plot(np.arange(nSteps),df1.surprisal_Metro[df1.Syncopation == 'Low'].iloc[0])
#ax1.set_ylabel('Surprisal', fontsize = 18)
ax1.set_title('Low ', fontsize = 18)
ax1.set_ylim(-0.2,2.3)
ax1.set_xlim(-2,130)

ax1.vlines(x = qbeat, ymin = -0.2, ymax =2.3, ls = '-', lw = 4, colors = 'lightgrey', zorder = 0)
ax1.vlines(x = ebeat, ymin = -0.2, ymax = 2.3, ls = '-',colors = 'lightgrey', zorder = 0)
ax1.vlines(x = mbeat, ymin = -0.2, ymax = 2.3, ls = '-',colors = 'grey', zorder = 0)

ax1.scatter( x = np.arange(nSteps),
              y = np.tile(df1.pattern3[df1.Syncopation == 'Low'],4), s = 10,c  = 'red')

ax1.scatter( x = np.arange(nSteps),
              y = np.tile(df1.Metronome[df1.Syncopation == 'Low'],4), s = 10,c  = 'green')

ax1.tick_params(axis='both',  pad=.002)

ax2.plot(np.arange(nSteps),df1.surprisal_Metro[df1.Syncopation == 'Medium'].iloc[0])

#ax2.set_ylabel('Surprisal', fontsize = 18)
ax2.set_title('Medium', fontsize = 18)
ax2.set_ylim(-0.2,2.3)
ax2.set_xlim(-2,130)

ax2.vlines(x = qbeat, ymin = -0.2, ymax = 2.3, ls = '-', lw = 4, colors = 'lightgrey', zorder = 0)
ax2.vlines(x = ebeat, ymin = -0.2, ymax = 2.3, ls = '-',colors = 'lightgrey', zorder = 0)
ax2.vlines(x = mbeat, ymin = -0.2, ymax = 2.3, ls = '-',colors = 'grey', zorder = 0)


ax2.scatter( x = np.arange(nSteps),
              y = np.tile(df1.pattern3[df1.Syncopation == 'Medium'],4), s = 10,c  = 'red')

ax2.scatter( x = np.arange(nSteps),
              y = np.tile(df1.Metronome[df1.Syncopation == 'Medium'],4), s = 10,c  = 'green')

ax2.tick_params(axis='both',  pad=.002)

ax3.plot(np.arange(nSteps),df1.surprisal_Metro[df1.Syncopation == 'High'].iloc[0])
ax3.set_title('High', fontsize = 18)
ax3.set_ylim(-0.2,2.3)
ax3.set_xlim(-2,130)

ax3.vlines(x = qbeat, ymin = -0.2, ymax = 2.3, ls = '-', lw = 4, colors = 'lightgrey', zorder = 0)
ax3.vlines(x = ebeat, ymin = -0.2, ymax = 2.3, ls = '-',colors = 'lightgrey', zorder = 0)
ax3.vlines(x = mbeat, ymin = -0.2, ymax = 2.3, ls = '-',colors = 'grey', zorder = 0)

ax3.scatter( x = np.arange(nSteps),
              y = np.tile(df1.pattern3[df1.Syncopation == 'High'],4), s = 10,c  = 'red')

ax3.scatter( x = np.arange(nSteps),
              y = np.tile(df1.Metronome[df1.Syncopation == 'High'],4), s = 10,c  = 'green')

ax3.tick_params(axis='both',  pad=.002)
ax3.set_xlabel('Time Step', fontsize = 16)

fig.tight_layout()
plt.savefig('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Manuscript\\Final Figures\\raw_posts_Surps\\surp_stimMet.png', dpi = 300)


#%% rhythm schematics

df_exploded = df_exploded.sort_values('syncIndex', ascending=True, ignore_index = True)

r1= sns.stripplot(data = df_exploded, x ='timeStep',
              y = df_exploded.pattern2,  size = 6, jitter = False,
              hue='Syncopation',palette="viridis")

#sns.set_theme(rc={'figure.figsize':(4,3.4)})
sns.despine(left = True, bottom = True)
sns.set_style('white')


r1.set(xlabel=None)
r1.set(ylabel=None)
r1.tick_params(left=False, labelsize = 12) 
r1.tick_params(bottom=False)
r1.set(yticklabels=[1,18,16,15,6,4,3,0,0,0]) 
r1.set(xticklabels = [])
r1.legend_.set_title(None)
plt.legend(loc="upper left", bbox_to_anchor=(.77, .65), fontsize = 12)
plt.ylim(0.8, nStim+.2)
plt.vlines(x = [0,8,16,24], ymin = 0.8, ymax = 9.2, ls = '-', lw = 4, colors = 'grey')
plt.vlines(x = [4,12,20,28], ymin = 0.8, ymax = 9.2, ls = '-',colors = 'grey')
plt.vlines(x = [2,6,10,14,18,22,26,30], ymin = 0.8, ymax = 9.2, ls = '-',lw = .5, colors = 'grey')
r1.grid(False)


plt.tight_layout()
plt.savefig('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Manuscript\\Final Figures\\TMrhythm_schem3.png', dpi = 300)
#plt.show()

#%% template plots

maxWeights = np.array([.9,.1,.3,.1,.5,.1,.3,.1,.7,.1,.3,.1,.5,.1, .3,.1,.8,.1,.3,.1,.5,.1,.3,.1,.7,.1,.3,.1,.5,.1,.3,.1 ])

fig, ax = plt.subplots(figsize = ( 5 , 4 ))

#fig = plt.figure()
#ax = fig.add_axes([0.1,0.1,0.75,0.75])
ax.bar(np.arange(len(maxWeights)), maxWeights)
ax.set_xticklabels([])
ax.set_yticklabels([])
#plt.title(r'Metered Template $\lambda_m(t)$', fontsize = 18)
plt.ylim(0,1)
#ax.set_ylabel('$P(onset)$', fontsize = 18)
#ax.set_xlabel('Time Step', fontsize = 18)
plt.tight_layout() 
#plt.show()
plt.savefig('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Manuscripts\\Annals\\Figures\\Pieces\\Graphical_abstract\\meterTemplate_clean.png', dpi = 300)



uniWeights = np.tile(mean(maxWeights), len(maxWeights))

fig, ax = plt.subplots(figsize = ( 5 , 4 ))
#uni weights
#fig = plt.figure()
#ax = fig.add_axes([0.1,0.1,0.75,0.75])
ax.bar(np.arange(len(maxWeights)),uniWeights)
ax.set_xticklabels([])
ax.set_yticklabels([])
#plt.title(r'Unmetered Template $\lambda_u(t)$', fontsize = 18)
#ax.set_ylabel('$P(onset)$', fontsize = 18)
#ax.set_xlabel('Time Step', fontsize = 18)
plt.ylim(0,1)
plt.tight_layout() 
#plt.show()
plt.savefig('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Manuscripts\\Annals\\Figures\\Pieces\\Graphical_abstract\\unmeterTemplate_clean.png', dpi = 300)


#%% raw posterior  without detail (for GA)
#from matplotlib.colors import ListedColormap
#my_cmap = ListedColormap(sns.color_palette('Set1', 3).as_hex())

cs = sns.color_palette('Set1', 3)

fig = plt.figure()
fig, ax = plt.subplots(figsize = ( 5 , 4 ))
#ax = fig.add_axes([0.1,0.1,0.75,0.75])
ax.plot(np.arange(len(dfm['postMet_nMetro'][3])),dfm['postMet_nMetro'][3], c = cs[0])
ax.plot(np.arange(len(dfm['postMet_Metro'][3])),dfm['postMet_Metro'][3], c = cs[1])
ax.set_xticklabels([])
ax.set_yticklabels([])
#plt.ylabel('Posterior', fontsize = 18)
#plt.xlabel('Time Step', fontsize = 18)
#plt.title('Low', fontsize = 24)
plt.ylim(0,1)
#plt.show()
plt.tight_layout()
plt.savefig('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Manuscripts\\Annals\\Figures\\Pieces\\Graphical_abstract\\rawPost_Medium.png', dpi = 300)

#%% raw surprisal  without detail (for GA)
fig = plt.figure()
fig, ax = plt.subplots(figsize = ( 5 , 4 ))
#ax = fig.add_axes([0.1,0.1,0.75,0.75])
ax.plot(np.arange(len(dfm['surprisal_nMetro'][7])),dfm['surprisal_nMetro'][7], c = cs[0])
ax.plot(np.arange(len(dfm['surprisal_Metro'][7])),dfm['surprisal_Metro'][7], c = cs[1])
ax.set_xticklabels([])
ax.set_yticklabels([])
#plt.ylabel('Surprisal', fontsize = 18)
#plt.xlabel('Time Step', fontsize = 18)
#plt.title('Low', fontsize = 24)
plt.ylim(0,2)
#plt.show()
plt.tight_layout()
plt.savefig('C:\\Users\\Tomas\\Dropbox\\Aarhus\\PIPPET\\Manuscripts\\Annals\\Figures\\Pieces\\Graphical_abstract\\rawSurp_High.png', dpi = 300)

