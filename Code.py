#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random as rand
from sklearn import preprocessing 


# In[75]:


pip install ControlBurn


# In[150]:


np.random.seed(26)
df = pd.read_csv('Data_F1.csv')

feature_names = df.columns
for i in range(len(feature_names)):
 print(str(i), "\t", str(feature_names[i]),"\t\t\t", str(type(df.iloc[0,i])))


# In[151]:


df.columns


# In[152]:


columns_to_norm = ['GP', 'MPG',
       'MIN%Minutes Percentage', 'USG%Usage Rate', 'TO%Turnover Rate', 'FTA',
       'FT','2PA', '2P', '3PA', '3P', 'eFG', 'TS', 'PPGPoints per game.',
       'RPGRebounds per game.', 'TRB%Total Rebound Percentage',
       'APGAssists per game.', 'Assist Percentage', 'SPGSteals per game.',
       'BPGBlocks per game.', 'TOPGTurnovers per game.', 'VI', 'ORTG', 'DRTG','WingSpan(cm)']


# In[153]:


scaler = preprocessing.StandardScaler().fit(df[columns_to_norm])


# In[154]:


N_Data = scaler.transform(df[columns_to_norm])


# In[155]:


print(N_Data)


# In[156]:


N_Data.astype


# In[157]:


df1 = pd.DataFrame(N_Data, columns = ['GP', 'MPG',
       'MIN%Minutes Percentage', 'USG%Usage Rate', 'TO%Turnover Rate', 'FTA',
       'FT','2PA', '2P', '3PA', '3P', 'eFG', 'TS', 'PPGPoints per game.',
       'RPGRebounds per game.', 'TRB%Total Rebound Percentage',
       'APGAssists per game.', 'Assist Percentage', 'SPGSteals per game.',
       'BPGBlocks per game.', 'TOPGTurnovers per game.', 'VI', 'ORTG', 'DRTG','WingSpan(cm)'])


# In[158]:


df1


# In[159]:


df1['Name']=df['FULL NAME']


# In[160]:


df1['POS']=df['POS']


# In[161]:


df1


# In[162]:


df1 = df1.dropna()
df1 = df1[~df['3P'].isna()] 
df1 = df1.sample(frac = 1) 
train_proportion = 0.6
n = len(df1)
print('Size of dataset: ', str(n))

t = int(train_proportion * n)
target = df1['3P']
target_columns = ['3PA','3P']
data = df1.loc[:, ~df1.columns.isin(target_columns)]
# the following variable records the features of examples in the training set
train_x = data.iloc[:t]
# the following variable records the features of examples in the test set
test_x = data.iloc[t:]
# the following variable records the labels of examples in the training set
train_y = target[:t]
# the following variable records the labels of examples in the test set
test_y = target[t:]

print('Training dataset: ', train_x)


# In[163]:


def string_to_float(string):
     try:
        return float(string)
     except:
        return 0.0
    
labels_real = [
 'GP',   
 'MPG',
 'WingSpan(cm)', 
 'FT',
 'VI',
 'PPGPoints per game.',
 'ORTG','2P',
 'RPGRebounds per game.',
 'TOPGTurnovers per game.',
 'BPGBlocks per game.',
 'SPGSteals per game.'
]

#labels_string = [
    #'POS',
    
#]


# In[164]:


train_vals_real = np.asarray(train_x[labels_real])

test_vals_real = np.asarray(test_x[labels_real])


# In[165]:


#assert(train_vals_from_string.applymap(lambda x:type(x)!=str).all(skipna=False).all(skipna=False) == True)
#assert(test_vals_from_string.applymap(lambda x:type(x)!=str).all(skipna=False).all(skipna=False) == True)


# In[166]:


#train_vals_from_string = np.asarray(train_vals_from_string)
#test_vals_from_string = np.asarray(test_vals_from_string)


# In[167]:


train_vals = np.concatenate((train_vals_real,np.ones((len(train_vals_real[:,0]),1))), axis = 1)

test_vals = np.concatenate((test_vals_real,np.ones((len(test_vals_real[:,0]),1))), axis = 1)


# In[168]:


def MSE(y, pred):
    return np.mean(np.power(np.subtract(y,pred),2)) 

# This function plots the main diagonal;for a "predicted vs true" plot with perfect predictions, all data lies on this line
def plotDiagonal(xmin, xmax):
    xsamples = np.arange(xmin,xmax,step=0.01)
    plt.plot(xsamples,xsamples,c='black')

# This helper function plots x vs y and labels the axes
def plotdata(x=None,y=None,xname=None,yname=None,margin=0.05,plotDiag=True,zeromin=False):
    plt.scatter(x,y,label='data')
    plt.xlabel(xname)
    plt.ylabel(yname)
    range_x = max(x) - min(x)
    range_y = max(y) - min(y)
    if plotDiag:
        plotDiagonal(min(x)-margin*range_x,max(x)+margin*range_x)
    if zeromin:
        plt.xlim(0.0,max(x)+margin*range_x)
        plt.ylim(0.0,max(y)+margin*range_y)
    else:
        plt.xlim(min(x)-margin*range_x,max(x)+margin*range_x)
        plt.ylim(min(y)-margin*range_y,max(y)+margin*range_y)
    plt.show()

# This function plots the predicted labels vs the actual labels (We only plot the first 1000 points to avoid slow plots)
def plot_pred_true(test_pred=None, test_y=None, max_points = 1000):
    plotdata(test_pred[1:max_points], test_y[1:max_points],'Predicted', 'True', zeromin=True)


# In[169]:


# This function runs OLS and bypasses any SVD (Singular Value Decomposition) convergence errors by refitting the model
def run_OLS(train_y, test_y, train_vals, test_vals):
    ols_model = sm.regression.linear_model.OLS(train_y, train_vals)
    while True: # Bypasses SVD convergence assertion error
        try:
            results = ols_model.fit()
            break
        except:
            None
            
    w = np.array(results.params).reshape([len(results.params),1])

    train_pred = np.matmul(train_vals,w)
    test_pred = np.matmul(test_vals,w)

    train_MSE = MSE(train_y, train_pred.flatten())
    test_MSE = MSE(test_y, test_pred.flatten())
    
    return train_MSE, test_MSE, test_pred,w


# In[170]:


train_MSE, test_MSE, test_pred,w = run_OLS(train_y, test_y, train_vals, test_vals)


# In[171]:


print("Train MSE\t", str(train_MSE))
print("Test MSE\t", str(test_MSE))

plot_pred_true(test_pred.flatten(), test_y) #.flatten() will make sure the dimensions match


# In[172]:


cat_labels = [
  'POS'
]

#Sets of all categories in a particular column
cats_sets = [train_x.loc[:, label].fillna('NaN').unique() for label in cat_labels]


# In[173]:


def onehot(column=None, col=None):
    column = column.astype(pd.CategoricalDtype(categories = col))
    hot_encode = pd.get_dummies(column)
    
    return np.asarray(hot_encode)


# In[174]:


trv = np.zeros((len(train_x),1))
for i in range(len(cat_labels)):
    trv = np.concatenate((trv,onehot(train_x[cat_labels[i]],cats_sets[i])),axis = 1)
     
train_cat_vals = trv[:,1:]

tv = np.zeros((len(test_x),1))
for i in range(len(cat_labels)):
    tv = np.concatenate((tv,onehot(test_x[cat_labels[i]],cats_sets[i])),axis = 1)
    
test_cat_vals = tv[:,1:]


# In[175]:


train_cat_vals.shape


# In[176]:


train_vals1 = np.concatenate((train_vals,train_cat_vals),axis = 1)
test_vals1 = np.concatenate((test_vals,test_cat_vals),axis = 1)


# In[177]:


train_vals1.shape


# In[178]:


train_ridge = np.concatenate((train_vals_real,train_cat_vals),axis = 1)


# In[179]:


test_ridge = np.concatenate((test_vals_real,test_cat_vals),axis = 1)


# In[180]:


train_MSE, test_MSE, test_pred,w = run_OLS(train_y, test_y, train_vals1, test_vals1)


# In[181]:


print("Train MSE\t", str(train_MSE))
print("Test MSE\t", str(test_MSE))

plot_pred_true(test_pred.flatten(), test_y)


# In[182]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge
alpha = .1 # regularization parameter

lin = Ridge(alpha=alpha).fit(train_ridge,train_y)
yhat = lin.predict(test_ridge)
yhat


# In[183]:


yhat.shape


# In[184]:


train_ridge.shape


# In[185]:


Ridge_MSE = MSE(test_y, yhat)
print(Ridge_MSE)


# In[186]:


plot_pred_true(yhat,test_y)


# In[187]:


pip install Mosek --user


# In[188]:


import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import mosek
import cvxpy as cp


# In[189]:


from ControlBurn.ControlBurn import ControlBurnRegressor
from ControlBurn.ControlBurn import ControlBurnClassifier

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error


# In[190]:


df3 = df[columns_to_norm]
df3 = df3.dropna()
df3 = df3[~df['3P'].isna()] 
df3 = df3.sample(frac = 1) 

target = df3['3P']
target_columns = ['3PA','3P']
data1 = df3.loc[:, ~df3.columns.isin(target_columns)]
# the following variable records the features of examples in the training set
train_x = data1.iloc[:]
# the following variable records the features of examples in the test set
# test_x = data1.iloc[t:]
# the following variable records the labels of examples in the training set
train_y = target[:]
# the following variable records the labels of examples in the test set
# test_y = target[t:]
print(train_x)


# In[191]:


print(train_y)


# In[192]:


X = train_x
y = train_y
y = pd.Series(y)
y.index = X.index
print(str(len(X)) + ' rows')
print(str(len(X.columns)) + ' columns')


# In[193]:


xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.33, random_state=42)
xTrainScaler = preprocessing.StandardScaler()
xTrain = xTrainScaler.fit_transform(xTrain)
xTrain = pd.DataFrame(xTrain,columns = X.columns)
xTest = preprocessing.StandardScaler().fit_transform(xTest)
xTest = pd.DataFrame(xTest,columns = X.columns)
yTrain = preprocessing.StandardScaler().fit_transform(yTrain.values.reshape(-1, 1))
yTest = preprocessing.StandardScaler().fit_transform(yTest.values.reshape(-1, 1))
yTrain = pd.Series(yTrain.flatten())
yTrain.index = xTrain.index
yTest = pd.Series(yTest.flatten())
yTest.index = xTest.index


# In[194]:


# xTrainScaler = preprocessing.StandardScaler()
# xTrain = xTrainScaler.fit_transform(train_ridge)
# xTrain = pd.DataFrame(xTrain,columns = .columns)


# In[195]:


cb = ControlBurnRegressor(build_forest_method = 'doublebagboost', alpha = 0.02)
cb.fit(xTrain,yTrain)
# cb.fit(train_vals1,test_vals1)
# print(cb.features_selected_) #print selected features
# print(cb.feature_importances_) #print feature importances

print('Number of trees grown: ' + str(len(cb.forest)))
print('Number of trees selected: ' + str(len(cb.subforest)))
print('Features selected ' + str(cb.features_selected_))


# In[228]:


tree_list = cb.subforest
cols = X.columns
print(cols)
# for tree in tree_list:
#     print(cols[tree.feature_importances_ > 0].values )


# In[226]:


sub_weights = cb.weights[cb.weights>0]
# print(sub_weights)
# print(sub_weights.shape)
for feat in cols:
    loc = 0
    pred_all = []
    for tree in tree_list:
        if ((feat in cols[tree.feature_importances_>0]) & (len(cols[tree.feature_importances_>0]) >= 1 )) :
            x_temp = pd.DataFrame(np.linspace(-1,1,1000),columns = [feat])
#             print(x_temp)
            for i in cols:
                if i != feat:
                    x_temp[i] = 0
            x_temp = x_temp[X.columns]
        
            pred = tree.predict(x_temp)
            pred_all.append(pred*sub_weights[loc])
        
        loc = loc+1
    pred_all = np.sum(pred_all,axis = 0)
    plt.plot(np.linspace(-1,1,1000),pred_all)
    plt.xlabel(feat)
    plt.ylabel('Contribution to Prediction')
    break


# In[198]:


import itertools
import seaborn as sns
from itertools import combinations,permutations
import matplotlib.pyplot as plt



pairs = list(permutations(cols,2))
counter = pd.DataFrame(pairs,columns = ['Feature 1','Feature 2'])

counts = []
for i in pairs:
    n = 0
    for tree in tree_list:
        feats = list(cols[tree.feature_importances_>0])
        if ((i[0] in feats) & (i[1] in feats)):
            n = n + 1
    counts.append(n)
counter['count'] = counts
counter = counter.pivot_table(index='Feature 1', columns='Feature 2', values='count')
mask = np.zeros_like(counter, dtype='bool')
mask[np.triu_indices_from(mask)] = True
sns.heatmap(counter, mask = mask , cmap = 'Blues')


# In[125]:


pairs = list(permutations(np.linspace(-3,3,200),2)) 
x_temp = pd.DataFrame(pairs,columns = ['eFG','WingSpan(cm)'])
for i in cols:
    if i not in ['eFG','WingSpan(cm)']:
        x_temp[i] = 0
x_temp = x_temp[X.columns]
pred_all = []
loc = 0
for tree in tree_list:
    if (('eFG' in cols[tree.feature_importances_>0]) &('WingSpan(cm)' in cols[tree.feature_importances_>0]) & (len(cols[tree.feature_importances_>0]) == 2)):
        pred = tree.predict(x_temp)
        pred_all.append(pred*sub_weights[loc])
        
    loc = loc + 1
pred_all = np.sum(pred_all,axis = 0)


df = pd.DataFrame(pairs,columns = ['eFG','WingSpan(cm)'])
df['contribution'] = pred_all

contribution = df['contribution']

#Unscale the data for easier interpretation
temp = df.drop('contribution',axis = 1)
for i in cols:
    if i not in ['eFG','WingSpan(cm)']:
        temp[i] = 0
temp = temp[X.columns]
temp = pd.DataFrame(xTrainScaler.inverse_transform(temp), columns = X.columns)
df = temp[['eFG','WingSpan(cm)']].round(3)
df['contribution'] = contribution.round(3)

df_plot = df.pivot_table(index='eFG', columns='WingSpan(cm)', values='contribution')
sns.heatmap(df_plot , cmap = 'RdBu', fmt='.4f')


# In[126]:


w,


# In[ ]:




