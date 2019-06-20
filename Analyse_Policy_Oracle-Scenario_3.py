# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import networkx as nx
import sys
import os

#choose the scenario you want to study
ScenarioStudied=3

# # Load the Score Table for all configurations

path_data = os.path.join('../results_MassiveSearch1/')
dataset_csv = os.path.join(path_data, "concat_rewards.csv")
rewards = pd.read_csv(dataset_csv, sep=",",)

ncolumns=rewards.shape[1]
rewards.head()

ncolumns

# +
#plot=rewards.iloc[:,2:ncolumns].sum().plot.bar()
# -

#plot the cumulated score for every configuration
plot2=rewards.iloc[:,2:ncolumns].sum().plot.hist()

# ## Reorder Reward DataFrame columns
# All configurations did not converge during the computations: only about 2200 over 8500. 
# Also they were computed in parallel so they are not ordered by their ID. That's what we do in the following.

# +
columnNames=rewards.iloc[:,2:ncolumns].columns
columnIndexes=[int(str.split(name,'_')[1]) for name in columnNames]

iDXOrder=np.argsort(columnIndexes,)
newColumnNames=columnNames[iDXOrder]
newColumnNames=['datetimes','scenario']+list(newColumnNames)
# -

rewards=rewards[newColumnNames]

sumRewardsConfig=rewards.iloc[:,2:ncolumns].sum()
idxHighRewards=np.where(sumRewardsConfig>=80000)

rewards

unique, counts = np.unique(rewards.iloc[:,2:ncolumns].idxmax(axis=1),return_counts=True)


print('here are the configurations which performed best at least once in the scenarios')
unique, counts 

"rewards_5832" in unique

# ## Histogram of times a configuration performed best

# +
# %matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go

# Connect Plotly in offline mode. 
init_notebook_mode(connected = True) 



data = [go.Bar(
        x=unique,
        y=counts
        )]

iplot(data)


# -

sumRewardsConfig[72]

sumRewardsMAx=rewards.iloc[:,2:ncolumns].max(axis=1).sum()

print("maximum possible reward over scenarios if you pick up the best configuration at each time, meaning you can play any action at each timestep")
sumRewardsMAx

#get the indices of the best configurations
iDXBest=np.argsort(-np.array(sumRewardsConfig))[0:50]
iDXBest

# # Load Table of Configuration description

path_data = os.path.join('../results_MassiveSearch1/')
actions_csv = os.path.join(path_data, "actions_table.csv")
actions = pd.read_csv(actions_csv, sep=",",)
actions=actions.drop(columns=['Unnamed: 0'])

# +

actions.head()
# -

actions.shape

# Filter the "relevant actions": the ones for which we have a score in the score table

#filter to only relevant actions
convergentActions=columnIndexes
convergentActions=np.sort(columnIndexes)
#convergentActions=convergentActions[1:10]
actionsRelevant=actions.iloc[convergentActions]
actionsRelevant=actionsRelevant.reset_index(drop=True)

convergentActions

actions.loc[convergentActions[iDXBest]]

# # Get Authorized Transitions
# Since you are only allowed one action to switch to a new configuration, this limits the configuration you can reach and transition to, from one timestep to another.
# Tha's this transition matrix we compute here

vectorReachableTopologies=[ [i] for i in range(actionsRelevant.shape[0]) ]

colmumnNames=np.array(actionsRelevant.columns)
#dfObj.duplicated(['Age', 'City'])

# To find the configurations one action away at most from a given configuration, you delete one action column in the action table and you find the similar rows. Then you loop over all columns.

# +
#on cherche les configurations qui sont a au plus une action de distance

for i in range(len(colmumnNames)):
    newCols=np.delete(colmumnNames,i)
    df=actionsRelevant[newCols]
    df = df[df.duplicated(keep=False)]
    df = np.array(df.groupby(df.columns.tolist()).apply(lambda x: tuple(x.index)).tolist())
    #print (df)

    for i in range(df.shape[0]):
        indices=np.array(df[i])
        for j in indices:
            #vectorReachableTopologies[j]=vectorReachableTopologies[j]+indices
            vectorReachableTopologies[j]=np.concatenate((vectorReachableTopologies[j],indices))
# -

vectorReachableTopologies=[np.unique(vectorReachableTopologies[j]) for j in range(len(vectorReachableTopologies))]

vectorReachableTopologies

# # Look at max rewards

rewards_scenario=rewards.loc[np.where(rewards['scenario']==ScenarioStudied)]

rewards_scenario

sumDoNothing=rewards_scenario['rewards_0'].sum()
sumDoNothing

rewardsValues=rewards_scenario.iloc[:,2:ncolumns].values
rewardsValues

# # Create graph Edges
# Over the length of a scenario, tou want to build a directed graph with every node being a configuration at some tiestep and every edge having the score of the configuration you are reaching.

# +
#vectorReachableTopologies=vectorReachableTopologies[0:10]
# -

# You build the adjacency edge list for you graph
# The source nodes are all the ones you can find in the score table with a timestep ID.
# The target nodes are all the configuration you can reach from the source node with a timestep ID of +1

# +
duration=int(rewards_scenario.shape[0]-1)
print("duration: "+ str(duration))

edgeNamesOr=[str(convergentActions[i])+'_' +str(int(t)) for t in range(duration) for i in range(len(vectorReachableTopologies)) 
             for j in vectorReachableTopologies[i] ]
edgeNamesEx=[str(convergentActions[j])+'_' +str(int(t+1)) for t in range(duration) for i in range(len(vectorReachableTopologies)) 
             for j in vectorReachableTopologies[i] ]
EdgeWeight=[rewardsValues[int(t+1),j] for t in range(duration) for i in range(len(vectorReachableTopologies)) 
             for j in vectorReachableTopologies[i] ]
# -

# You add an init node at the beginning of your graph and an 'end' node at the end to make one single connected graph with a global source node and global target node to then compute the best path between those

edgeNamesOrNodeSource=['init' for j in vectorReachableTopologies]
edgeNamesExNodeSource=[str(convergentActions[i])+'_' +str(0) for i in range(len(vectorReachableTopologies))]
EdgeWeightNodeSource=[rewardsValues[0,j] for j in range(len(vectorReachableTopologies))]

edgeNamesExNodeEnd=['end' for j in vectorReachableTopologies]
edgeNamesOrNodeEnd=[str(convergentActions[i])+'_' +str(int(duration)) for i in range(len(vectorReachableTopologies))]
EdgeWeightNodeEnd=[0.1 for j in vectorReachableTopologies]

edgeNamesOr=edgeNamesOrNodeSource+edgeNamesOr+edgeNamesOrNodeEnd
edgeNamesEx=edgeNamesExNodeSource+edgeNamesEx+edgeNamesExNodeEnd
EdgeWeight=EdgeWeightNodeSource+EdgeWeight+EdgeWeightNodeEnd

edgeDf=pd.DataFrame({'or':edgeNamesOr,'ex':edgeNamesEx,'weight':EdgeWeight})

edgeDf.head()

np.where(edgeDf['or']=='1_2')

# # Build the graph
# You build a directed graph to compute then a longest path (not a shortest path!)

G=nx.from_pandas_edgelist(edgeDf, target='ex', source='or', edge_attr=['weight'],create_using=nx.DiGraph())

# +
#G.edges()
#G.nodes()
#G.get_edge_data('8_0','8_1',default=0)

# +
#G.nodes()

# +
#sortedComps=sorted(nx.strongly_connected_components(G), key=len, reverse=True)

# +
#len(sortedComps)

# +
#nx.draw(G, with_labels = True)
# -

# ## Get the shortest path for an acyclic directed graph

longestPath=nx.dag_longest_path(G)
longestPath

# +
#totalWeight=0
#for i in range(len(longestPath)-1):
#    sourceNode=longestPath[i]
#    targetNode=longestPath[i+1]
#    weight=G[sourceNode][targetNode]['weight']
#    print("new edge")
#    print(sourceNode)
#    print(targetNode)
#    print(weight)
#    totalWeight+=weight
#
# -

# Make an histogram of the most used configurations over the shortest path

configsShortest=['reward_'+str.split(name,'_')[0] for name in longestPath[1:-1]]

unique, counts = np.unique(configsShortest,return_counts=True)

unique

# +
# %matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go

# Connect Plotly in offline mode. 
init_notebook_mode(connected = True) 



data = [go.Bar(
        x=unique,
        y=counts
        )]

iplot(data)


# -

# # Compare the cumulated scores for some baselines
# - Do Nothing in Ref Topo
# - Do Nothing in the best Topoes
# - Best path with only one action at each timestep
# - Best path with all actions possible

print('max reward by doin:')
nx.dag_longest_path_length(G)

sumDoNothinginRef=rewards_scenario[['rewards_0']].sum()
print('max reward by doing nothing in reference configuration:')
print(sumDoNothinginRef)


# +
sums=rewards_scenario.iloc[:,2:ncolumns].sum()
sumDoNothinginBest=sums.max()

print('max reward by doing nothing in a configuration:')
print(sums.argmax())
print(sumDoNothinginBest)
# -

sumRewardsMAx=rewards_scenario.iloc[:,2:ncolumns].max(axis=1).sum()
print('max reward by doing the best configuration with all actions possible:')
sumRewardsMAx

plot2=rewards_scenario.iloc[:,2:ncolumns].sum().plot.hist()

idxActions=[17,25,41,56,57,72,88,585,601,617,826,56,650]
actions.iloc[idxActions]

actions.iloc[5832]
