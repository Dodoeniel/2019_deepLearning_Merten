"""library for handling documentation with pickle files"""
import pandas as pd
import os

# Pickle Functions
def readDataFromPickle(config, filename):
    """Loads Data from Pickle File"""
    path = config.picklePath + filename

    # if stopid in pickle variable and it is a dataframe-> sort columns
    return pd.read_pickle(path)

def writeDataToPickle(variable, path):
    """Writes a given variable to a file"""
    directory = os.path.dirname(path)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    if os.path.isfile(path):
        raise FileExistsError(path)
    else:
        variable.to_pickle(path)

def assertLabelAssignmentIsUnchanged(X, y):
    """Checks, if order of ids is identical in X and y

    Background Info: Pickle files change order of stopids due to different sort algorithm
    This function ensures that identical sort algorithm was applied to X and y

    X: DataFrame with one of the following format:

        index = 'stopID':
            (index)    f1(quantity1)  f2(quantity1)
            (stopId)
                       ----------------------------
            1.1501       ...            ...
            2.1501       ...            ...
            3.1501       ...            ...

        column = 'stopID':
            (index)  stopId  time quantity1 quantity2
                     --------------------------------
              0      1.1501   1     ...       ...
              1      1.1501   2     ...       ...
              2      1.1501   3     ...       ...
              3      2.1501   1     ...       ...
              4      2.1501   2     ...       ...

    y:
        index = 'stopID':
            (index)    labels
            (stopId)
                        ------------
             1.1501       ...
             2.1501       ...
             3.1501       ...
    """

    # Get List of StopIds from X
    if X.index.name == 'stopId':
        stopIdsInX = X.index.tolist()

    elif 'stopId' in X.columns:
        stopIdsInX = []
        for stopId, data in X.groupby('stopId', sort=False):
            stopIdsInX.append(stopId)
            # Check if time is still in coorect order
            if 'time' in data.columns:
                currentOrderOfTime = data['time']
                sortedOrderOfTime  = data['time'].sort()
                assert(currentOrderOfTime == sortedOrderOfTime)
    else:
        raise ValueError('X does not contain information on stopIds')

    # Get List of StopIds from labels
    stopIdsinLabels = y.index.tolist()

    # Ensure order is identical
    assert(stopIdsInX == stopIdsinLabels)