# Start By reading the csv from the DATA URL
- We can load a csv from a url using pandas' read_csv function
- The CSV will be stored as a pandas DataFrame object

### Note on notation
- names that should be treated as constants are ALL CAPS


```python
import pandas

# import numpy
import numpy as np

#import typing modules for type annotations
from typing import List, Dict, Callable, Tuple

DATA_URL = 'https://raw.githubusercontent.com/mpourhoma/CS4661/master/iris.csv'
IRIS_DATA_SET = pandas.read_csv(DATA_URL)

print(IRIS_DATA_SET)
```

         sepal_length  sepal_width  petal_length  petal_width    species
    0             5.1          3.5           1.4          0.2     setosa
    1             4.9          3.0           1.4          0.2     setosa
    2             4.7          3.2           1.3          0.2     setosa
    3             4.6          3.1           1.5          0.2     setosa
    4             5.0          3.6           1.4          0.2     setosa
    ..            ...          ...           ...          ...        ...
    145           6.7          3.0           5.2          2.3  virginica
    146           6.3          2.5           5.0          1.9  virginica
    147           6.5          3.0           5.2          2.0  virginica
    148           6.2          3.4           5.4          2.3  virginica
    149           5.9          3.0           5.1          1.8  virginica
    
    [150 rows x 5 columns]


# Next we split our data set into a training and testing data set
- here our test data set is 40% of the values and our random state is set to 10, think of this like a random seed


```python
from sklearn.model_selection import train_test_split

TRAIN, TEST = train_test_split(IRIS_DATA_SET, test_size=0.4, random_state=10)

print(f'\nTraining Data Set:\n\n {TRAIN}\n\nTesting Data Set:\n\n {TEST}')
```

    
    Training Data Set:
    
          sepal_length  sepal_width  petal_length  petal_width     species
    29            4.7          3.2           1.6          0.2      setosa
    3             4.6          3.1           1.5          0.2      setosa
    106           4.9          2.5           4.5          1.7   virginica
    98            5.1          2.5           3.0          1.1  versicolor
    140           6.7          3.1           5.6          2.4   virginica
    ..            ...          ...           ...          ...         ...
    113           5.7          2.5           5.0          2.0   virginica
    64            5.6          2.9           3.6          1.3  versicolor
    15            5.7          4.4           1.5          0.4      setosa
    125           7.2          3.2           6.0          1.8   virginica
    9             4.9          3.1           1.5          0.1      setosa
    
    [90 rows x 5 columns]
    
    Testing Data Set:
    
          sepal_length  sepal_width  petal_length  petal_width     species
    87            6.3          2.3           4.4          1.3  versicolor
    111           6.4          2.7           5.3          1.9   virginica
    10            5.4          3.7           1.5          0.2      setosa
    91            6.1          3.0           4.6          1.4  versicolor
    49            5.0          3.3           1.4          0.2      setosa
    60            5.0          2.0           3.5          1.0  versicolor
    72            6.3          2.5           4.9          1.5  versicolor
    67            5.8          2.7           4.1          1.0  versicolor
    39            5.1          3.4           1.5          0.2      setosa
    55            5.7          2.8           4.5          1.3  versicolor
    66            5.6          3.0           4.5          1.5  versicolor
    142           5.8          2.7           5.1          1.9   virginica
    53            5.5          2.3           4.0          1.3  versicolor
    1             4.9          3.0           1.4          0.2      setosa
    19            5.1          3.8           1.5          0.3      setosa
    112           6.8          3.0           5.5          2.1   virginica
    85            6.0          3.4           4.5          1.6  versicolor
    38            4.4          3.0           1.3          0.2      setosa
    21            5.1          3.7           1.5          0.4      setosa
    35            5.0          3.2           1.2          0.2      setosa
    102           7.1          3.0           5.9          2.1   virginica
    132           6.4          2.8           5.6          2.2   virginica
    126           6.2          2.8           4.8          1.8   virginica
    24            4.8          3.4           1.9          0.2      setosa
    61            5.9          3.0           4.2          1.5  versicolor
    2             4.7          3.2           1.3          0.2      setosa
    95            5.7          3.0           4.2          1.2  versicolor
    90            5.5          2.6           4.4          1.2  versicolor
    76            6.8          2.8           4.8          1.4  versicolor
    117           7.7          3.8           6.7          2.2   virginica
    58            6.6          2.9           4.6          1.3  versicolor
    97            6.2          2.9           4.3          1.3  versicolor
    129           7.2          3.0           5.8          1.6   virginica
    114           5.8          2.8           5.1          2.4   virginica
    146           6.3          2.5           5.0          1.9   virginica
    47            4.6          3.2           1.4          0.2      setosa
    124           6.7          3.3           5.7          2.1   virginica
    120           6.9          3.2           5.7          2.3   virginica
    118           7.7          2.6           6.9          2.3   virginica
    141           6.9          3.1           5.1          2.3   virginica
    26            5.0          3.4           1.6          0.4      setosa
    43            5.0          3.5           1.6          0.6      setosa
    59            5.2          2.7           3.9          1.4  versicolor
    41            4.5          2.3           1.3          0.3      setosa
    56            6.3          3.3           4.7          1.6  versicolor
    32            5.2          4.1           1.5          0.1      setosa
    52            6.9          3.1           4.9          1.5  versicolor
    70            5.9          3.2           4.8          1.8  versicolor
    121           5.6          2.8           4.9          2.0   virginica
    144           6.7          3.3           5.7          2.5   virginica
    68            6.2          2.2           4.5          1.5  versicolor
    109           7.2          3.6           6.1          2.5   virginica
    81            5.5          2.4           3.7          1.0  versicolor
    78            6.0          2.9           4.5          1.5  versicolor
    51            6.4          3.2           4.5          1.5  versicolor
    14            5.8          4.0           1.2          0.2      setosa
    48            5.3          3.7           1.5          0.2      setosa
    63            6.1          2.9           4.7          1.4  versicolor
    20            5.4          3.4           1.7          0.2      setosa
    137           6.4          3.1           5.5          1.8   virginica


# Now we define a function called get_knn_classifiers
- This function will return a list of Knn classifier objects based on an iterable of k values provided


```python
from sklearn.neighbors import KNeighborsClassifier

def get_knn_classifiers(k_values: List[int]) -> List[KNeighborsClassifier]:
    return [KNeighborsClassifier(n_neighbors=k_value) for k_value in k_values]

```

# Now we can define 2 functions, fit_and_predict and get_y_predictions_multiple
- fit and predict with take a knn classifier, fit it to a training data set, then return a prediction on test data
- get_y_predictions_multiple_k uses fit_and_predict in a list comprehension to create a list of tuples (prediction, k_value)


```python
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
```

    [(array(['versicolor', 'virginica', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'virginica', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'virginica', 'setosa',
           'versicolor', 'virginica', 'virginica', 'virginica', 'virginica',
           'virginica', 'versicolor', 'versicolor', 'versicolor', 'setosa',
           'setosa', 'versicolor', 'setosa', 'virginica'], dtype=object), 1), (array(['versicolor', 'virginica', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'versicolor', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'virginica', 'virginica', 'virginica', 'versicolor',
           'virginica', 'versicolor', 'versicolor', 'versicolor', 'setosa',
           'setosa', 'virginica', 'setosa', 'virginica'], dtype=object), 5), (array(['versicolor', 'virginica', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'versicolor', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'virginica', 'virginica', 'virginica', 'versicolor',
           'virginica', 'versicolor', 'versicolor', 'versicolor', 'setosa',
           'setosa', 'versicolor', 'setosa', 'virginica'], dtype=object), 7), (array(['versicolor', 'virginica', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'versicolor', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'versicolor', 'setosa',
           'virginica', 'virginica', 'virginica', 'virginica', 'versicolor',
           'virginica', 'versicolor', 'versicolor', 'versicolor', 'setosa',
           'setosa', 'versicolor', 'setosa', 'virginica'], dtype=object), 15), (array(['versicolor', 'virginica', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'versicolor', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'versicolor', 'setosa',
           'virginica', 'versicolor', 'virginica', 'virginica', 'versicolor',
           'virginica', 'versicolor', 'versicolor', 'versicolor', 'setosa',
           'setosa', 'versicolor', 'setosa', 'virginica'], dtype=object), 27), (array(['virginica', 'virginica', 'setosa', 'virginica', 'setosa',
           'versicolor', 'virginica', 'virginica', 'setosa', 'virginica',
           'virginica', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'virginica', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'virginica', 'setosa',
           'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
           'virginica', 'versicolor', 'virginica', 'virginica', 'setosa',
           'setosa', 'virginica', 'setosa', 'virginica'], dtype=object), 59)]


# Let's get some metrics!

- Now we define a function called get_accuracy_scores
- This will allow us to get how accurate we were on our prediction results. This function takes into account the tuple we pass it for labeling purposes down the line

## Results
- We can see that a higher k value does not necessarily mean a higher prediction accuracy
- In this case a k value of 7 gives the highest accuracy

k value of 7 produces ~97% ! pretty good so far


```python
from sklearn.metrics import accuracy_score

def get_accuracy_scores(test_y, y_predictions):
    return { additional_value : accuracy_score(test_y, y_predict) for y_predict, additional_value in y_predictions}

print(get_accuracy_scores(TEST['species'], Y_PREDICT_MULTIPLE_K))
```

    {1: 0.9166666666666666, 5: 0.95, 7: 0.9666666666666667, 15: 0.95, 27: 0.95, 59: 0.6666666666666666}


# Now Lets See which Single Feature Gives the Highest Accuracy
- here we define a function called get_y_single_feature_predictions
- this function uses a list comprehension to produce a list of tuples of the single feature data sets we are looking for
- feature name is included in the tuple for easy labeling purposes
- now we can get all the single feature predicitions for the data set by calling fit_and_predict like before
- here we have to reshape the series into (-1, 1) arrays since we are doing single feature data sets and need consistent 2 dimensional arrays rather than Series


```python
def get_y_single_feature_predictions():
    single_feature_sets = [ (TRAIN[feature], TEST[feature], feature) for feature in FEATURE_COLUMNS]
    return [ (fit_and_predict(KNeighborsClassifier(n_neighbors=11),\
                                                 train_col.values.reshape(-1, 1),\
                                                 TRAIN['species'], test_col.values.reshape(-1, 1)), feature)\
                                for train_col, test_col, feature in single_feature_sets]

Y_SINGLE_FEATURE_PREDICTIONS = get_y_single_feature_predictions()

print(Y_SINGLE_FEATURE_PREDICTIONS)
```

    [(array(['virginica', 'virginica', 'versicolor', 'virginica', 'setosa',
           'setosa', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'versicolor', 'versicolor', 'setosa', 'setosa',
           'virginica', 'virginica', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'virginica',
           'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica',
           'virginica', 'virginica', 'virginica', 'versicolor', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'setosa', 'setosa', 'virginica', 'setosa',
           'virginica', 'virginica', 'versicolor', 'virginica', 'virginica',
           'virginica', 'versicolor', 'virginica', 'virginica', 'versicolor',
           'setosa', 'virginica', 'versicolor', 'virginica'], dtype=object), 'sepal_length'), (array(['versicolor', 'virginica', 'setosa', 'virginica', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'setosa', 'versicolor',
           'virginica', 'virginica', 'versicolor', 'virginica', 'setosa',
           'virginica', 'setosa', 'virginica', 'setosa', 'virginica',
           'virginica', 'versicolor', 'versicolor', 'setosa', 'virginica',
           'virginica', 'virginica', 'versicolor', 'versicolor', 'setosa',
           'versicolor', 'versicolor', 'virginica', 'versicolor',
           'versicolor', 'virginica', 'virginica', 'virginica', 'versicolor',
           'setosa', 'setosa', 'setosa', 'virginica', 'versicolor',
           'virginica', 'setosa', 'setosa', 'virginica', 'versicolor',
           'virginica', 'versicolor', 'setosa', 'versicolor', 'versicolor',
           'virginica', 'setosa', 'setosa', 'versicolor', 'setosa', 'setosa'],
          dtype=object), 'sepal_width'), (array(['versicolor', 'virginica', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'versicolor', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'versicolor', 'setosa',
           'virginica', 'virginica', 'virginica', 'virginica', 'versicolor',
           'virginica', 'versicolor', 'versicolor', 'versicolor', 'setosa',
           'setosa', 'versicolor', 'setosa', 'virginica'], dtype=object), 'petal_length'), (array(['versicolor', 'virginica', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'versicolor', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'virginica', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'virginica', 'setosa',
           'versicolor', 'virginica', 'virginica', 'virginica', 'versicolor',
           'virginica', 'versicolor', 'versicolor', 'versicolor', 'setosa',
           'setosa', 'versicolor', 'setosa', 'virginica'], dtype=object), 'petal_width')]


# What did we get?

- looks like petal_width and petal_length produce the best accuracies independently


```python
SINGLE_FEATURE_SCORES = get_accuracy_scores(TEST['species'], Y_SINGLE_FEATURE_PREDICTIONS)

print(sorted(SINGLE_FEATURE_SCORES.items(), key=lambda x: x[1], reverse=True))

```

    [('petal_width', 0.95), ('petal_length', 0.9333333333333333), ('sepal_length', 0.6166666666666667), ('sepal_width', 0.5333333333333333)]


# Now for Doubles
- for getting the best combinations of double features, the process is very similar to single features
## I love itertools <3
- with itertools we can use the combinations function to produce all possible n sized pairs from a list
- this will give a list of tuples, which can later be cast to lists for indexing our datasets
- since we are now dealing with full DataFrames we don't have to worry about reshaping like we did with single features


```python
from itertools import combinations

def get_y_double_feature_predictions():
    double_features = list(combinations(FEATURE_COLUMNS, 2))
    return [(fit_and_predict(KNeighborsClassifier(n_neighbors=11), TRAIN[list(double_feature)], TRAIN['species'], TEST[list(double_feature)]), double_feature) for double_feature in double_features]
    
Y_DOUBLE_FEATURE_PREDICTIONS = get_y_double_feature_predictions()

print(Y_DOUBLE_FEATURE_PREDICTIONS)
```

    [(array(['virginica', 'virginica', 'setosa', 'virginica', 'setosa',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'versicolor', 'versicolor', 'setosa', 'setosa',
           'virginica', 'virginica', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica',
           'versicolor', 'virginica', 'virginica', 'versicolor', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'virginica', 'setosa',
           'virginica', 'virginica', 'versicolor', 'virginica', 'virginica',
           'virginica', 'versicolor', 'virginica', 'virginica', 'setosa',
           'setosa', 'virginica', 'setosa', 'virginica'], dtype=object), ('sepal_length', 'sepal_width')), (array(['versicolor', 'virginica', 'setosa', 'virginica', 'setosa',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'versicolor', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'virginica', 'setosa',
           'versicolor', 'virginica', 'virginica', 'virginica', 'versicolor',
           'virginica', 'versicolor', 'versicolor', 'versicolor', 'setosa',
           'setosa', 'virginica', 'setosa', 'virginica'], dtype=object), ('sepal_length', 'petal_length')), (array(['versicolor', 'virginica', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'virginica', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'virginica', 'setosa',
           'versicolor', 'virginica', 'virginica', 'virginica', 'virginica',
           'virginica', 'versicolor', 'virginica', 'versicolor', 'setosa',
           'setosa', 'versicolor', 'setosa', 'virginica'], dtype=object), ('sepal_length', 'petal_width')), (array(['versicolor', 'virginica', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'versicolor', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'versicolor', 'setosa',
           'virginica', 'versicolor', 'virginica', 'virginica', 'versicolor',
           'virginica', 'versicolor', 'versicolor', 'versicolor', 'setosa',
           'setosa', 'versicolor', 'setosa', 'virginica'], dtype=object), ('sepal_width', 'petal_length')), (array(['versicolor', 'virginica', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'versicolor', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'virginica', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'virginica', 'virginica', 'virginica', 'versicolor',
           'virginica', 'versicolor', 'versicolor', 'versicolor', 'setosa',
           'setosa', 'versicolor', 'setosa', 'virginica'], dtype=object), ('sepal_width', 'petal_width')), (array(['versicolor', 'virginica', 'setosa', 'versicolor', 'setosa',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'versicolor',
           'versicolor', 'virginica', 'versicolor', 'setosa', 'setosa',
           'virginica', 'versicolor', 'setosa', 'setosa', 'setosa',
           'virginica', 'virginica', 'virginica', 'setosa', 'versicolor',
           'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica',
           'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica',
           'setosa', 'virginica', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'versicolor', 'setosa', 'versicolor', 'setosa',
           'virginica', 'virginica', 'virginica', 'virginica', 'versicolor',
           'virginica', 'versicolor', 'versicolor', 'versicolor', 'setosa',
           'setosa', 'versicolor', 'setosa', 'virginica'], dtype=object), ('petal_length', 'petal_width'))]


# What did we get?
- looks like (sepal_width, petal_width produces the highest accuracy)

### These features are not what we found in the ranking of single features!
- Single features may produce a higher accuracy independently, but that does not guarentee a higher accuracy when paired
- pairing 2 features that have low accuracy on their own may produce an information gain higher than that produced by choosing the most accurate features independently.
- petal width and petal length still were quite close to the lead, but in the end were beat our by the combination of sepal_width and petal_width


```python
DOUBLE_FEATURE_SCORES = get_accuracy_scores(TEST['species'], Y_DOUBLE_FEATURE_PREDICTIONS)

print(sorted(DOUBLE_FEATURE_SCORES.items(), key=lambda x: x[1], reverse=True))
```

    [(('sepal_width', 'petal_width'), 0.9666666666666667), (('sepal_width', 'petal_length'), 0.95), (('petal_length', 'petal_width'), 0.95), (('sepal_length', 'petal_length'), 0.9166666666666666), (('sepal_length', 'petal_width'), 0.9), (('sepal_length', 'sepal_width'), 0.7333333333333333)]



```python

```
