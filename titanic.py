# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

import xgboost as xgb

import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

# loads our dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
dataset = [train, test]
PassengerId = test['PassengerId']
# see info on our dataset
train.info()
print('_'*40)
test.info()

# drop Cabin
test = test.drop(['Cabin'], axis=1)
train = train.drop(['Cabin'], axis=1)
dataset = [train, test]

# visualiza dataset
for column in train.columns.values:
    if column == 'Survived': continue
    print(train[[column, 'Survived']].groupby([column], as_index=False).mean().sort_values(by='Survived', ascending=False))

# drop ticket
test = test.drop(['Ticket'], axis=1)
train = train.drop(['Ticket'], axis=1)
dataset = [train, test]

#visualize age data on histogram
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# look at peoples titles and create a new column called titles
for data in dataset:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])

for data in dataset:
    # substitute the titles with low occurency for a Rare title
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    # Merge equivalent titles
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

# see our new title column    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# map the title tags into numbers so we can use it
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for data in dataset:
    # substitute the titles for our mapping
    data['Title'] = data['Title'].map(title_mapping)
    # fill Null and NaN values by 0
    data['Title'] = data['Title'].fillna(0)

# look at our new head
#train.head()

# drop the Name column and PassengerId of the training set
test = test.drop(['Name'], axis=1)
train = train.drop(['Name', 'PassengerId'], axis=1)
dataset = [train, test]
# look at our dataset shape
#test.shape, train.shape

# map the Sex tags into integers so we can use it
for data in dataset:
    # substitute the sexes for our mapping
    data['Sex'] = data['Sex'].map({'male':0, 'female':1}).astype(int)

# map the Embarked tags into integers so we can use it
# get the most frequent port
freq_port = train.Embarked.dropna().mode()[0]
for data in dataset:
    # fill null info with the most frequent port
    data['Embarked'] = data['Embarked'].fillna(freq_port)
    # substitute the letters for our mapping
    data['Embarked'] = data['Embarked'].map({'C':0, 'Q':1, 'S':2}).astype(int)

# plot the age of people with the same sex and PClass
#grid = sns.FacetGrid(train, row='Pclass', col='Sex', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend()

# substitute the null ages by the mean of main demographics that we have
guess_ages = np.zeros((2,3))
for data in dataset:
    for i in range(0, 2):
        for j in range(0, 3):
            # get the mean
            guess_df = data[(data['Sex'] == i) & \
                                  (data['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            # substitute null values by the mean
            data.loc[ (data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    # assert type as int
    data['Age'] = data['Age'].astype(int)

# create a age band column, so we can know how to group our age data
train['AgeBand'] = pd.cut(train['Age'], 5)
# Show our ageband
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# substitute age for our age band
for data in dataset:    
    data.loc[ data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age'] = 4

# drop the age band column
train = train.drop(['AgeBand'], axis=1)

# look at our data 
#train.head()
#train.tail()

# there is a missing value on our test data, so lets find its index so we can find a good value for it using its sex and PClass
pd.isnull(test).any(1).nonzero()[0] # returns 152
test.iloc[[152]] # we can see that its a male of Pclass 3

# make a guess of the fare
guess_df = data[(data['Sex'] == 0) & \
                                  (data['Pclass'] == 3)]['Fare'].dropna()

# substitute the null value for our mean
fare_guess = guess_df.median()
test.loc[ (data.Fare.isnull()), 'Fare'] = fare_guess

# now let's create a fare band, like the age band we did before (look for pd.cut x pd.qcut)
train['FareBand'] = pd.cut(train['Fare'], 5)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

# substitute Fare for fare band and drop FareBand after
for data in dataset:    
    data.loc[ data['Fare'] <= 103, 'Fare'] = 0
    data.loc[(data['Fare'] > 103) & (data['Fare'] <= 205), 'Fare'] = 1
    data.loc[(data['Fare'] > 205) & (data['Fare'] <= 308), 'Fare'] = 2
    data.loc[ data['Fare'] > 308, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)
    
train = train.drop(['FareBand'], axis=1)

# fare wasn't substitute in the train set, so we do it again
train.loc[ train['Fare'] <= 103, 'Fare'] = 0
train.loc[(train['Fare'] > 103) & (train['Fare'] <= 205), 'Fare'] = 1
train.loc[(train['Fare'] > 205) & (train['Fare'] <= 308), 'Fare']   = 2
train.loc[ train['Fare'] > 308, 'Fare'] = 3
train['Fare'] = train['Fare'].astype(int)

# get our X and Y for the training set
y_train = train['Survived']
train = train.drop("Survived", axis=1)
test  = test.drop("PassengerId", axis=1).copy()



vote_est = [
    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc',ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),

    #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    
    #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ('lr', linear_model.LogisticRegressionCV()),
    
    #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    
    #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
    ('knn', neighbors.KNeighborsClassifier()),
    
    #SVM: http://scikit-learn.org/stable/modules/svm.html
    ('svc', svm.SVC(probability=True)),
    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    ('xgb', xgb.XGBClassifier()),
    #multi-layer perceptron: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    ('mlp', MLPClassifier())
]

#WARNING: Running is very computational intensive and time expensive.
#Code is written for experimental/developmental purposes and not production ready!

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
#Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_n_estimator = [100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .05, .1]
grid_max_depth = [2, 4, 6]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['entropy']
grid_bool = [True, False]
grid_seed = [0]


grid_param = [
            [{
            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
            'n_estimators': grid_n_estimator, #default=50
            'learning_rate': grid_learn, #default=1
            #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
            'random_state': grid_seed
            }],
       
    
            [{
            #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
            'n_estimators': grid_n_estimator, #default=10
            'max_samples': grid_ratio, #default=1.0
            'random_state': grid_seed
             }],

    
            [{
            #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
            'n_estimators': grid_n_estimator, #default=10
            'criterion': grid_criterion, #default=”gini”
            'max_depth': grid_max_depth, #default=None
            'random_state': grid_seed
             }],


            [{
            #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
            #'loss': ['deviance', 'exponential'], #default=’deviance’
            'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
            'max_depth': grid_max_depth, #default=3   
            'random_state': grid_seed
             }],

    
            [{
            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
            'n_estimators': grid_n_estimator, #default=10
            'criterion': grid_criterion, #default=”gini”
            'max_depth': grid_max_depth, #default=None
            'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
            'random_state': grid_seed
             }],
    
            [{    
            #GaussianProcessClassifier
            'max_iter_predict': grid_n_estimator, #default: 100
            'random_state': grid_seed
            }],
        
    
            [{
            #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
            'fit_intercept': grid_bool, #default: True
            #'penalty': ['l1','l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs
            'random_state': grid_seed
             }],
            
    
            [{
            #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
            'alpha': grid_ratio, #default: 1.0
             }],
    
    
            #GaussianNB - 
            [{}],
    
            [{
            #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
            'n_neighbors': [1,2,3,4,5,6,7], #default: 5
            'weights': ['uniform', 'distance'], #default = ‘uniform’
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }],
            
    
            [{
            #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
            #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
            #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [1,2,3,4,5], #default=1.0
            'gamma': grid_ratio, #edfault: auto
            'decision_function_shape': ['ovo', 'ovr'], #default:ovr
            'probability': [True],
            'random_state': grid_seed
             }],

    
            [{
            #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
            'learning_rate': grid_learn, #default: .3
            'max_depth': [1,2,4,6,8,10], #default 2
            'n_estimators': grid_n_estimator, 
            'seed': grid_seed  
             }],

            
            [{
            #multi-layer perceptron - http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
            'solver':['lbfgs'],
            'alpha':[1e-5],
            'hidden_layer_sizes':[(5, 2)],
            'random_state':[1]
            }]
        ]


for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip

    #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm

      
    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc')
    best_search.fit(train, y_train)
    best_param = best_search.best_params_
    print('The best parameter for {} is {}'.format(clf[1].__class__.__name__, best_param))
    clf[1].set_params(**best_param) 


print('-'*10)

# Use hard voting to make our predictions
grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
grid_hard_cv = model_selection.cross_validate(grid_hard, train, y_train, cv  = cv_split)
grid_hard.fit(train, y_train)
predictions = grid_hard.predict(test)

# Use soft voting to make our predictions
'''
grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
grid_soft_cv = model_selection.cross_validate(grid_soft, train, y_train, cv  = cv_split)
grid_soft.fit(train, y_train)
predictions = grid_soft.predict(test)
'''

# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)
