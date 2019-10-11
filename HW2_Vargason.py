#!/usr/bin/env python
# coding: utf-8

# # Start By reading the csv from the DATA URL
# - We can load a csv from a url using pandas' read_csv function
# - The CSV will be stored as a pandas DataFrame object
# 
# ### Note on notation
# - names that should be treated as constants are ALL CAPS

# In[344]:


import pandas

# import numpy
import numpy as np

#import typing modules for type annotations
from typing import List, Dict, Callable, Tuple

DATA_URL = 'https://raw.githubusercontent.com/mpourhoma/CS4661/master/iris.csv'
IRIS_DATA_SET = pandas.read_csv(DATA_URL)

print(IRIS_DATA_SET)


# # Next we split our data set into a training and testing data set
# - here our test data set is 40% of the values and our random state is set to 10, think of this like a random seed

# In[345]:


from sklearn.model_selection import train_test_split

TRAIN, TEST = train_test_split(IRIS_DATA_SET, test_size=0.4, random_state=10)

print(f'\nTraining Data Set:\n\n {TRAIN}\n\nTesting Data Set:\n\n {TEST}')


# # Now we define a function called get_knn_classifiers
# - This function will return a list of Knn classifier objects based on an iterable of k values provided

# In[346]:


from sklearn.neighbors import KNeighborsClassifier

def get_knn_classifiers(k_values: List[int]) -> List[KNeighborsClassifier]:
    return [KNeighborsClassifier(n_neighbors=k_value) for k_value in k_values]


# # Now we can define 2 functions, fit_and_predict and get_y_predictions_multiple
# - fit and predict with take a knn classifier, fit it to a training data set, then return a prediction on test data
# - get_y_predictions_multiple_k uses fit_and_predict in a list comprehension to create a list of tuples (prediction, k_value)

# In[347]:


FEATURE_COLUMNS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def fit_and_predict(knn: KNeighborsClassifier, train_data: pandas.DataFrame, train_target: pandas.Series , test_data: pandas.DataFrame) -> Tuple[np.array, int]:
    knn.fit(train_data, train_target)
    return knn.predict(test_data)

def get_y_predictions_multiple_k() -> List[Tuple[np.array, int]]:
    train_X = TRAIN[FEATURE_COLUMNS]
    train_y = TRAIN['species']
    test_X = TEST[FEATURE_COLUMNS]
    return [(fit_and_predict(knn, train_X, train_y, test_X), knn.n_neighbors) for knn in get_knn_classifiers([1, 5, 7, 15, 27, 59])]

Y_PREDICT_MULTIPLE_K = get_y_predictions_multiple_k()

print(Y_PREDICT_MULTIPLE_K)


# # Let's get some metrics!
# 
# - Now we define a function called get_accuracy_scores
# - This will allow us to get how accurate we were on our prediction results. This function takes into account the tuple we pass it for labeling purposes down the line
# 
# ## Results
# - We can see that a higher k value does not necessarily mean a higher prediction accuracy
# - In this case a k value of 7 gives the highest accuracy
# 
# k value of 7 produces ~97% ! pretty good so far

# In[348]:


from sklearn.metrics import accuracy_score

def get_accuracy_scores(test_y, y_predictions):
    return { additional_value : accuracy_score(test_y, y_predict) for y_predict, additional_value in y_predictions}

print(get_accuracy_scores(TEST['species'], Y_PREDICT_MULTIPLE_K))


# # Now Lets See which Single Feature Gives the Highest Accuracy
# - here we define a function called get_y_single_feature_predictions
# - this function uses a list comprehension to produce a list of tuples of the single feature data sets we are looking for
# - feature name is included in the tuple for easy labeling purposes
# - now we can get all the single feature predicitions for the data set by calling fit_and_predict like before
# - here we have to reshape the series into (-1, 1) arrays since we are doing single feature data sets and need consistent 2 dimensional arrays rather than Series

# In[349]:


def get_y_single_feature_predictions():
    single_feature_sets = [ (TRAIN[feature], TEST[feature], feature) for feature in FEATURE_COLUMNS]
    return [ (fit_and_predict(KNeighborsClassifier(n_neighbors=11),                                                 train_col.values.reshape(-1, 1),                                                 TRAIN['species'], test_col.values.reshape(-1, 1)), feature)                                for train_col, test_col, feature in single_feature_sets]

Y_SINGLE_FEATURE_PREDICTIONS = get_y_single_feature_predictions()

print(Y_SINGLE_FEATURE_PREDICTIONS)


# # What did we get?
# 
# - looks like petal_width and petal_length produce the best accuracies independently

# In[350]:


SINGLE_FEATURE_SCORES = get_accuracy_scores(TEST['species'], Y_SINGLE_FEATURE_PREDICTIONS)

print(sorted(SINGLE_FEATURE_SCORES.items(), key=lambda x: x[1], reverse=True))


# # Now for Doubles
# - for getting the best combinations of double features, the process is very similar to single features
# ## I love itertools <3
# - with itertools we can use the combinations function to produce all possible n sized pairs from a list
# - this will give a list of tuples, which can later be cast to lists for indexing our datasets
# - since we are now dealing with full DataFrames we don't have to worry about reshaping like we did with single features

# In[351]:


from itertools import combinations

def get_y_double_feature_predictions():
    double_features = list(combinations(FEATURE_COLUMNS, 2))
    return [(fit_and_predict(KNeighborsClassifier(n_neighbors=11), TRAIN[list(double_feature)], TRAIN['species'], TEST[list(double_feature)]), double_feature) for double_feature in double_features]
    
Y_DOUBLE_FEATURE_PREDICTIONS = get_y_double_feature_predictions()

print(Y_DOUBLE_FEATURE_PREDICTIONS)


# # What did we get?
# - looks like (sepal_width, petal_width produces the highest accuracy)
# 
# ### These features are not what we found in the ranking of single features!
# - Single features may produce a higher accuracy independently, but that does not guarentee a higher accuracy when paired
# - pairing 2 features that have low accuracy on their own may produce an information gain higher than that produced by choosing the most accurate features independently.
# - petal width and petal length still were quite close to the lead, but in the end were beat our by the combination of sepal_width and petal_width

# In[352]:


DOUBLE_FEATURE_SCORES = get_accuracy_scores(TEST['species'], Y_DOUBLE_FEATURE_PREDICTIONS)

print(sorted(DOUBLE_FEATURE_SCORES.items(), key=lambda x: x[1], reverse=True))


# In[ ]:




