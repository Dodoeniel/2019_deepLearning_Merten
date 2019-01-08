"""library for data preprocessing steps for time series

@author: Nadine

# features erzeugen
# smoothen ( time series bearbeiten)
# normalizieren ( time series oder feauters)
# outlier detection...
# upsampling
"""

import random
import collections
import pandas as pd
import numpy as np
import math
import random
from sklearn.preprocessing import MinMaxScaler

#import tsfresh
#import tsfresh.utilities.dataframe_functions as tsfreshUtil

import logging


# tsFresh based feature generation
def generateFeaturesFromTimeSeries(X_ts):
    """
    Parameters:
        X_ts: Flat DataFrame
            Contains time series Data from which features should be extracted
            Example:
            (index)  stopId  time quantity1 quantity2
                     --------------------------------
              0      1.1501   1     ...       ...
              1      1.1501   2     ...       ..
              2      1.1501   3     ...       ...
              3      2.1501   1     ...       ...
              4      2.1501   2     ...       ...

    Returns:
        X_feat: DataFrame
            Contains extracted Features and their values

            (index)    f1(quantity1)  f2(quantity1)  f1(quantity2)  f3(quantity2)
            (stopId)
                       ----------------------------------------------------------
            1.1501       ...            ...            ...            ...
            2.1501       ...            ...            ...            ...
            3.1501       ...            ...            ...            ...
    """

    X_feat = tsfresh.extract_features(X_ts, column_id="stopId", column_sort="time", n_jobs=12)

    # Replac NaN- Values with 0
    tsfreshUtil.impute(X_feat)
    # Replac index 'id' with 'stopId'
    X_feat.index.name = "stopId"

    return X_feat

def selectRelevantFeatures(X_feat, labels):
    """
    Parameters:
        X_feat: DataFrame
            Contains extracted Features and their values

            (index)    f1(quantity1)  f2(quantity1)  f1(quantity2)  f3(quantity2)
                       ----------------------------------------------------------
            1.1501       ...            ...            ...            ...
            2.1501       ...            ...            ...            ...
            3.1501       ...            ...            ...            ...

        labels: Series
            Contains labels for time series Data
                0: No Squeal, 1: Squeal

            (index)  label
                     -----
             1.1501    0
             2.1501    0
             3.1501    1
             4.1501    1
             5.1501    0

    Returns:
        X_feat_relevant: DataFrame
            Contains relevant features and their values

            (index)    f1(quantity1)  f1(quantity2)  f3(quantity2)
                       -------------------------------------------
            1.1501       ...            ...            ...
            2.1501       ...            ...            ...
            3.1501       ...            ...            ...
    """
    X_feat_relevant = tsfresh.select_features(X_feat, labels)

    summarizeFeatureSelection(X_feat, X_feat_relevant)

    numberOfRelevantFeatures = X_feat_relevant.shape[1]
    if numberOfRelevantFeatures == 0:
        logging.error("Did not find relevant features.")
        #raise ValueError("Did not find relevant features.")

    return X_feat_relevant

def getFeatureNames(X_feat):
    """
    Parameters:
        X_feat: DataFrame
            Contains extracted Features and their values

            (index)    f1(quantity1)  f2(quantity1)  f1(quantity2)  f3(quantity2)
                       ----------------------------------------------------------
            1.1501       ...            ...            ...            ...
            2.1501       ...            ...            ...            ...
            3.1501       ...            ...            ...            ...

    Returns:
        featureNames: List
            Contains names of all Features stored in X_feat
    """

    featureNames = X_feat.columns.tolist()
    return featureNames

def getSubsetOfFeatures(X_feat, featuresToInclude):

    X_feat_subset = X_feat[featuresToInclude]

    return  X_feat_subset


# Splitting of Data
def splitData(X, labels, val_split = 0.2, test_split = 0.2):

    assert(val_split + test_split < 1)

    numOfTotalExmpls   = X.shape[0]
    numOfExmplsInVal   = round(val_split  * numOfTotalExmpls)
    numOfExmplsInTest  = round(test_split * numOfTotalExmpls)
    numOfExmplsInTrain = numOfTotalExmpls - numOfExmplsInVal - numOfExmplsInTest

    assert(numOfExmplsInTrain > 0)

    X_shuffled, y_shuffled = shuffleData(X, labels)

    X_train = X_shuffled[:numOfExmplsInTrain]
    y_train = y_shuffled[:numOfExmplsInTrain]

    X_test  = X_shuffled[numOfExmplsInTrain:(numOfExmplsInTrain + numOfExmplsInTest)]
    y_test  = y_shuffled[numOfExmplsInTrain:(numOfExmplsInTrain + numOfExmplsInTest)]

    X_val   = X_shuffled[(numOfExmplsInTrain + numOfExmplsInTest):]
    y_val   = y_shuffled[(numOfExmplsInTrain + numOfExmplsInTest):]

    # Create Classes: collections.namedtuple(classname, constructor)
    TrainData = collections.namedtuple('trainData', ['X', 'labels'])
    TestData  = collections.namedtuple('testData',  ['X', 'labels'])
    ValData   = collections.namedtuple('valData',   ['X', 'labels'])

    # Instanciate Classes
    trainData = TrainData(X_train, y_train)
    testData  = TestData(X_test, y_test)
    valData   = ValData(X_val, y_val)

    summarizeDataSplitting(trainData, testData, valData)

    return trainData, testData, valData

def shuffleData(X, labels):
    """X: DataFrame with unique indices (bzw wird zeilenweise gesplittet, nicht fuer flatDF geeignet)"""

    assert(sorted(X.index.tolist()) == sorted(labels.index.tolist()))

    indices = X.index.tolist()
    random.shuffle(indices)
    X_shuffled = X.loc[indices]
    labels_shuffled = labels.loc[indices]

    return X_shuffled, labels_shuffled


# Balancing of Data
def getBalancedDataByDropping(X, labels, targetPercent, reference):

    X.sort_index(inplace=True)
    labels.sort_index(inplace=True)

    assert (X.index.tolist() == labels.index.tolist())
    assert (targetPercent <=100)
    # annahme: Squeal ist unterbesetzt

    X_label_0 = X[labels == 0]
    y_label_0 = labels[labels == 0]

    X_label_1 = X[labels == 1]
    y_label_1 = labels[labels == 1]

    (X_label_0_shuffled, y_label_0_shuffled) = shuffleData(X_label_0, y_label_0)
    (X_label_1_shuffled, y_label_1_shuffled) = shuffleData(X_label_1, y_label_1)

    numOfSquealExamples    = len(X_label_1)
    numOfNoSquealsExamples = len(X_label_0)

    if(reference == 'Squeal'):

        numOfNoSquealsExamples = round(numOfSquealExamples * (100-targetPercent) / targetPercent)
        X_0_new = X_label_0_shuffled[:numOfNoSquealsExamples]
        y_0_new = y_label_0_shuffled[:numOfNoSquealsExamples]

        X_balanced = pd.concat([X_label_1, X_0_new])
        y_balanced = pd.concat([y_label_1, y_0_new])

    elif(reference == 'NoSqueal'):

        numOfSquealExamples = round(numOfNoSquealsExamples * (100-targetPercent) / targetPercent)
        X_1_new = X_label_1_shuffled[:numOfSquealExamples]
        y_1_new = X_label_1_shuffled[:numOfSquealExamples]

        X_balanced = pd.concat([X_label_0, X_1_new])
        y_balanced = pd.concat([y_label_0, y_1_new])

    else:
        X_balanced = None
        y_balanced = None

    summarizeBalancing(labels, y_balanced)
    return X_balanced, y_balanced


# Data Normalization
def normalizeFeatureDatasetsScaledByStd(X_feat_train, X_feat_test, X_feat_val):
        meanOfTrain = X_feat_train.mean()
        stdOfTrain  = X_feat_train.std()

        X_feat_train_normalized = normalizeFeaturesScaledByStd(X_feat_train, meanOfTrain, stdOfTrain)
        X_feat_test_normalized  = normalizeFeaturesScaledByStd(X_feat_test, meanOfTrain, stdOfTrain)
        X_feat_val_normalized   = normalizeFeaturesScaledByStd(X_feat_val, meanOfTrain, stdOfTrain)

        return X_feat_train_normalized, X_feat_test_normalized, X_feat_val_normalized

def normalizeFeaturesScaledByStd(X_feat, mean, std):
    """ Scaling and mean normalization of features using std deviation

    Parameters:
        X_feat: DataFrame
            Contains original values.
            One row per example and one column per feature

            (index)   f1   f2   f3
                    -----------------
            1.1051   ...  ...  ...
            2.1051   ...  ...  ...
            3.1051   ...  ...  ...

        mean: Series
            Contains mean for each feature

            (index)  mean
                    ------
               f1     ..
               f2     ..
               f3     ..

        std: Series
            Contains standard deviation for each feature

            (index)  std
                    ------
               f1     ..
               f2     ..
               f3     ..

    Returns:
        X_feat_normalized: DataFrame
            Contains normalized values.
            One row per example and one column per feature.

            (index)   f1   f2   f3
                    -----------------
            1.1051   ...  ...  ...
            2.1051   ...  ...  ...
            3.1051   ...  ...  ...
    """

    std[std == 0] = 1  # avoid to divide by zero
    X_feat_normalized = (X_feat - mean)/std

    return X_feat_normalized

def normalizeFeatureDatasetsScaledByRange(X_feat_train, X_feat_test, X_feat_val):
    maxOfTrain  = X_feat_train.max()
    minOfTrain  = X_feat_train.min()
    meanOfTrain = X_feat_train.mean()

    X_feat_train_normalized = normalizeFeaturesScaledByRange(X_feat_train, meanOfTrain, maxOfTrain, minOfTrain)
    X_feat_test_normalized  = normalizeFeaturesScaledByRange(X_feat_test,  meanOfTrain, maxOfTrain, minOfTrain)
    X_feat_val_normalized   = normalizeFeaturesScaledByRange(X_feat_val,   meanOfTrain, maxOfTrain, minOfTrain)

    return X_feat_train_normalized, X_feat_test_normalized, X_feat_val_normalized

def normalizeFeaturesScaledByRange(X_feat, mean, max, min):
    """ Scaling and mean normalization of features using range

    Parameters:
        X_feat: DataFrame
            Contains original values.
            One row per example and one column per feature

            (index)   f1   f2   f3
                    -----------------
            1.1051   ...  ...  ...
            2.1051   ...  ...  ...
            3.1051   ...  ...  ...

        mean: Series
            Contains mean for each feature

            (index)  mean
                    ------
               f1     ..
               f2     ..
               f3     ..

        max: Series
            Contains max value for each feature

            (index)  max
                    ------
               f1     ..
               f2     ..
               f3     ..

        min: Series
            Contains min value for each feature

            (index)  min
                    ------
               f1     ..
               f2     ..
               f3     ..

    Returns:
        X_feat_normalized: DataFrame
            Contains normalized values.
            One row per example and one column per feature.

            (index)   f1   f2   f3
                    -----------------
            1.1051   ...  ...  ...
            2.1051   ...  ...  ...
            3.1051   ...  ...  ...
    """
    valueRange = (max-min)
    valueRange[valueRange == 0] = 1  # avoid to divide by zero
    X_feat_normalized = (X_feat - mean)/valueRange

    return X_feat_normalized


# Smoothing of Data
def smoothingEedData(X_ts):

    smoothedEedData = []
    for stopId, eedDataOfStopId in X_ts.groupby('stopId', sort=False):
        smoothedEedDataOfStopId = smoothingEedDataForStopId(eedDataOfStopId)
        smoothedEedData.append(smoothedEedDataOfStopId)

    X_ts_smoothed = pd.concat(smoothedEedData)

    return X_ts_smoothed

def smoothingEedDataForStopId(eedDataOfStopId):
    """smoothes Data from one eed file"""

    smoothedEedDataForStopId = []
    excludedColumns = ['stopId', 'time']

    for columnname in eedDataOfStopId.columns:

        column = eedDataOfStopId[columnname]

        if columnname not in excludedColumns:
            windowSize     = getSmoothingWindowSizeForColumn(columnname)
            columnSmoothed = smoothingSeries(column, windowSize)
            smoothedEedDataForStopId.append(columnSmoothed)
        else:
            smoothedEedDataForStopId.append(column)

    # Create Data Frame from Smoothed Columns
    entriesOfDict   = [(series.name, series) for series in smoothedEedDataForStopId] # [(key1, val1), (key2, val2),..]
    eedDataAsDict   = collections.OrderedDict(entriesOfDict)
    smoothedEedData = pd.DataFrame.from_dict(eedDataAsDict)

    return smoothedEedData

def smoothingSeries(column, rollingWindowsize):
    # center = true -> sonst versatz
    # min period = 1 -> enden keine NaN values
    column_smooth = column.rolling(rollingWindowsize, min_periods=1, center=True).mean()
    return column_smooth

def getSmoothingWindowSizeForColumn(columnname):
    defaultWindowSize = 3
    windowSizesForColumn = {
        'p1': 3,
        'n1': 3
    }
    if columnname in windowSizesForColumn:
        windowSize = windowSizesForColumn[columnname]
    else:
        windowSize = defaultWindowSize

    return  windowSize


# Daniels functions
def dropDataChannels(X_ts, channel_names):
    """
    Drops specific columns from the data set
    :param X_ts: all operational parameter data Nadines Format X_ts
    @author: Daniel
    :param channel_names: vector of not wanted channels by name
    :return: X_ts with omitted columns
    """
    return(X_ts.drop(channel_names, axis=1))


def balanceDataDaniel(X_ts, labels, target):
    """
    @author Daniel
    Compact version of balancing data sets
    :param X_ts: Flat Dataframe of all Data
    :param labels: Flat Dataframe of labels during preprocessing
    :param target: percentage of squealing data sets
    :return: balanced X_ts and labels
    """
    nrOnes = sum(labels.values)     # calc number of data sets that squeal
    nrZeros = len(labels.values) - nrOnes   # calc number of data sets that do not squeal
    drop_nr = nrZeros - (math.floor(nrOnes*100/target) - nrOnes)    # calc number of data sets to drop
    while drop_nr > 0:
        # select random data set by index = stopId
        indexToDrop = random.choice(labels.index.tolist())
        # check whether that da set squeals
        if labels.get(indexToDrop) == 0:
            # drop data set from labels
            labels = labels.drop(labels=indexToDrop)
            # create new X_ts with respective stopId being not part of that
            X_ts = X_ts[X_ts.stopId != indexToDrop]
            drop_nr -= 1 # reduce drop nr by one

    return X_ts, labels

def balanceSlicedData(X_ts, labels_sliced, target, distributed_Output = True, COLUMN_ID = 'sliceId'):
    labels_single = pd.Series()
    # iterate over all unique IDs
    for id in labels_sliced[COLUMN_ID].unique():
        # for each id the maximum of the labels is searched, resulting in either 0 or 1
        maximum = max(labels_sliced.loc[labels_sliced[COLUMN_ID] == id, 'label'])
        labels_single = labels_single.append(pd.Series([maximum], index=[id]))
    nrOnes = sum(labels_single.values)  # calc number of data sets that squeal
    nrZeros = len(labels_single.values) - nrOnes  # calc number of data sets that do not squeal
    drop_nr = nrZeros - (math.floor(nrOnes * 100 / target) - nrOnes)  # calc number of data sets to drop

    while drop_nr > 0:
        # select random data set by index = stopId
        indexToDrop = random.choice(labels_single.index.tolist())
        # check whether that da set squeals
        if labels_single.get(indexToDrop) == 0:
            # drop data set from labels
            labels_single = labels_single.drop(labels=indexToDrop)
            labels_sliced = labels_sliced[labels_sliced[COLUMN_ID] != indexToDrop]
            # create new X_ts with respective stopId being not part of that
            X_ts = X_ts[X_ts[COLUMN_ID] != indexToDrop]

            drop_nr -= 1  # reduce drop nr by one

    if distributed_Output:
        return X_ts, labels_sliced
    else:
        return X_ts, labels_single


def splitDataPandasFormat(X_ts, labels, split, COLUMN_ID = 'stopId'):
    labels_single = pd.Series()
    X_test = pd.DataFrame()
    y_test = pd.DataFrame()
    for id in labels[COLUMN_ID].unique():
        # for each id the maximum of the labels is searched, resulting in either 0 or 1
        maximum = max(labels.loc[labels[COLUMN_ID] == id, 'label'])
        labels_single = labels_single.append(pd.Series([maximum], index=[id]))
    nrOnes = sum(labels_single.values)  # calc number of data sets that squeal

    for i in range(math.floor(split*nrOnes)):
        id = random.choice(labels_single.index.tolist())
        init_squeal = bool(labels_single[id])
        X_test = X_test.append(X_ts[X_ts[COLUMN_ID] == id], ignore_index=False)
        y_test = y_test.append(labels[labels[COLUMN_ID] == id], ignore_index=False)
        # remove id
        labels_single = labels_single.drop(labels=id)
        X_ts = X_ts[X_ts[COLUMN_ID] != id]
        labels = labels[labels[COLUMN_ID] != id]

        ## get exactly one squealing or not squaling data set to keep balancing intact
        while True:
            id = random.choice(labels_single.index.tolist())
            balance_squeal = bool(labels_single[id])

            if (init_squeal & (not balance_squeal)) or (not init_squeal & (balance_squeal)):
                X_test = X_test.append(X_ts[X_ts[COLUMN_ID] == id], ignore_index=False)
                y_test = y_test.append(labels[labels[COLUMN_ID] == id], ignore_index=False)
                # remove id
                labels_single = labels_single.drop(labels=id)
                X_ts = X_ts[X_ts[COLUMN_ID] != id]
                labels = labels[labels[COLUMN_ID] != id]
                break
    return (X_ts, labels), (X_test, y_test)


def getTimeDistributedLabels(eec_data, X_ts):
    """
    @author: Daniel
    extends the labels towards time distributedLabels,
    flag for type of labeling: 0,1
    :param eec_data: eec data as loaded by eec_csv_to_eecData
    :param X_ts: X_ts as flat Dataframe
    :return: Flat data frame with time and stopId containing which time step is associated with squealing (1) and not (0)
    """
    try:    # get labels
        labels = pd.DataFrame({'stopId': X_ts['stopId'],
                               'time': X_ts['time'],
                               'label': np.zeros((len(X_ts['stopId']),))})
    except KeyError:
        print('KeyError intercepted. \n Make sure stopId is included in the X_ts data')

    for i in eec_data.index: # iterate over eec_date to find different StopIds
        stopId = eec_data.get_value(i, 'stopId') # get associated stopId
        index_eec = eec_data.index[eec_data['stopId'] == stopId].tolist()[0] # get associated index
        if not np.isnan(eec_data.get_value(index_eec, 'd_1')): # check whether squealing occured
            # iterate all possible squealing
            nrSqueals = 0
            while True:
                nrSqueals += 1 # index of squeals in eec data begins with 1
                # check whether squealing occurs
                if not np.isnan(eec_data.get_value(index_eec, 'd_'+str(nrSqueals))):
                    # get squealing start and stop time
                    start = eec_data.get_value(index_eec, 'time_' + str(nrSqueals) + '_start')
                    stop = eec_data.get_value(index_eec, 'time_' + str(nrSqueals) + '_end')
                else:
                    # break if no squealing occurs
                    break
                # find indexes of time steps with noise in it
                labels.loc[(labels['stopId'] == stopId) & (labels['time'] >= start) & (labels['time'] <= stop), 'label'] = 1
    return labels


def windowData_all(X_ts, labels, w_length, hop_size, discard=True):
    """
    @author: Daniel
    Hops with a window over all data and returns windowed data
    :param X_ts: Data in Nadines data format
    :param labels: time distributed labels
    :param w_length: window length in seconds
    :param hop_size: hop size in seconds
    :param discard: Boolean, discard to small signals and remains
    :return: windowed Data and labels
    """
    # pre define output
    X_sliced = pd.DataFrame()
    labels_sliced = pd.DataFrame()

    # transform w_length into number of steps
    # TODO very hacky and not concurrent with later approaches
    # get time step difference
    diff = (X_ts.get_value(1, 'time') - X_ts.get_value(0, 'time'))[0]
    # calculate integer of how many time steps are needed to fulfill w_length
    w_length = math.floor(w_length/diff)
    hop_size = math.floor(hop_size/diff)
    for stopId in X_ts['stopId'].unique():
        # get the current series that shall be sliced
        # loc function used as access to the data frame
        X_snippet = X_ts.loc[X_ts['stopId'] == stopId]
        label_snippet = labels.loc[labels['stopId'] == stopId]
        # check length of signals
        if len(X_snippet) >= w_length or (len(X_snippet) < w_length and not discard):
            hop = 0 # number of hops done
            for hop in range((len(X_snippet)-w_length)//hop_size):
                # create Ids for sliced TS:  1.051_0 .... 1.051_0 1.051_1 .... 1.051_1
                sliceId = []
                for i in range(w_length):
                    sliceId.append(stopId + '_' + str(hop))

                # slice labels
                curr_label = label_snippet[hop * hop_size:hop * hop_size + w_length]
                curr_label.index = range(w_length)
                curr_label['sliceId'] = pd.Series(sliceId, index=curr_label.index)
                labels_sliced = pd.concat([labels_sliced, curr_label])

                # slice data
                curr_X = X_snippet[hop * hop_size:hop * hop_size + w_length]
                curr_X.index = range(w_length)
                curr_X['sliceId'] = pd.Series(sliceId, index=curr_X.index)
                X_sliced = pd.concat([X_sliced, curr_X])

            # remainder of windowed function or window bigger than original signal
            if (((len(X_snippet)-w_length) % hop_size > 0) & (not discard)) and ((len(X_snippet) < w_length) & (not discard)):
                if hop != 0:
                    hop += 1
                    curr_label = label_snippet[hop * hop_size + w_length:]
                    curr_X = X_snippet[hop * hop_size + w_length:]
                else:
                    curr_label = label_snippet
                    curr_X = X_snippet

                sliceId = stopId + '_' + str(hop)
                curr_label.index = range(len(curr_label))
                zero_padding = pd.DataFrame(np.zeros(((w_length - len(curr_label)), curr_label.shape[1])),
                                            index=range(len(curr_label), w_length),
                                            columns=curr_label.columns.values.tolist())
                curr_label = pd.concat([curr_label, zero_padding])
                curr_label['sliceId'] = pd.Series(sliceId, index=curr_label.index)
                labels_sliced = pd.concat([labels_sliced, curr_label])

                # remaining data
                curr_X.index = range(len(curr_X))

                zero_padding = pd.DataFrame(np.zeros(((w_length - len(curr_X)), curr_X.shape[1])),
                                            index=range(len(curr_X), w_length),
                                            columns=curr_X.columns.values.tolist())
                curr_X = pd.concat([curr_X, zero_padding])
                curr_X['sliceId'] = pd.Series(sliceId, index=curr_X.index)
                X_sliced = pd.concat([X_sliced, curr_X])
    return X_sliced, labels_sliced


def removeTime_single(X_ts, labels, truncate, stopId):
    """
    TODO Obsolete
    :param X_ts:
    :param labels:
    :param truncate:
    :param stopId:
    :return:
    """
    # get the one series that shall be truncated
    X_snippet = X_ts.loc[X_ts['stopId'] == stopId]
    label_snippet = labels.loc[labels['stopId'] == stopId]
    middleTime = X_snippet.get_value(len(X_snippet) // 2, 'time')
    X_snippet = X_snippet.drop(np.where((X_snippet['time'] < truncate[0])
                                        | (X_snippet['time'] > X_snippet.get_value(len(X_snippet) - 1, 'time') - truncate[2])
                                        | ((X_snippet['time'] >= middleTime - 0.5 * truncate[1])
                                           & (X_snippet['time'] < middleTime + 0.5 * truncate[1])))[0])
    label_snippet = label_snippet.drop(np.where((label_snippet['time'] < truncate[0])
                                        | (label_snippet['time'] > label_snippet.get_value(len(label_snippet) - 1, 'time') - truncate[2])
                                        | ((label_snippet['time'] >= middleTime - 0.5 * truncate[1])
                                           & (label_snippet['time'] < middleTime + 0.5 * truncate[1])))[0])
    return X_snippet, label_snippet


def removeTime_all(X_ts, labels, truncate):
    """
    TODO Obsolete
    :param X_ts:
    :param labels:
    :param truncate:
    :return:
    """
    X_snippet = pd.DataFrame()
    labels_snippet = pd.DataFrame()
    for stopId in X_ts['stopId'].unique():
        X_single, label_single = removeTime_single(X_ts, labels, truncate, stopId)
        X_snippet = pd.concat([X_snippet, X_single])
        labels_snippet = pd.concat([labels_snippet, label_single])
    return X_snippet, labels_snippet


def truncate_single(X_single, label_single, duration, location, discard):
    """
    @author truncation of a single time series for usage in truncate_all
    :param X_single:
    :param label_single:
    :param duration:
    :param location:
    :param discard:
    :return:
    """
    CENTER = 'center'
    FIRST = 'first'
    LAST = 'last'
    X_truncated = pd.DataFrame()
    labels_truncated = pd.DataFrame()
    diff = X_single.get_value(1, 'time') - X_single.get_value(0, 'time')
    nrOfSteps = math.floor(duration / diff)
    ### do truncation based on flag
    if location == CENTER:
        middleIndex = X_single.first_valid_index() + ((X_single.last_valid_index() - X_single.first_valid_index())//2)
        middleTime = X_single.get_value(middleIndex, 'time')
        startIndex = X_single.loc[X_single['time'] <= middleTime-0.5*duration].last_valid_index()
        if startIndex is None:
            startIndex = X_single.first_valid_index()
        elif startIndex > 0:  # a little bit hacky but improves robustness for cases where the lenght equals the duration
            startIndex -= 1

        X_truncated = X_single.iloc[startIndex:startIndex+nrOfSteps, :]
        labels_truncated = label_single.iloc[startIndex:startIndex + nrOfSteps, :]
    elif location == FIRST:
        X_truncated = X_single.loc[X_single['time'] <= duration] # TODO do with steps
        labels_truncated = label_single.loc[label_single['time'] <= duration]
    elif location == LAST:
        X_truncated = X_single.loc[X_single['time'] >= X_single.get_value(X_single.last_valid_index(), 'time') - duration]
        labels_truncated = label_single.loc[label_single['time'] >= label_single.get_value(label_single.last_valid_index(), 'time') - duration]

    # check whether time series is long enough for duration
    if X_truncated.shape[0] == nrOfSteps:
        return X_truncated, labels_truncated
    # if it is to short and discard flag is set to zero --> zero padding
    elif not discard:
        while True:
            zero_padding = pd.DataFrame(np.zeros((1, X_truncated.shape[1])), columns=X_truncated.columns.values.tolist())
            zero_padding = zero_padding.set_value(0, 'stopId', X_single.get_value(0, 'stopId'))
            X_truncated = X_truncated.append(zero_padding, ignore_index=True)
            zero_padding = pd.DataFrame(np.zeros((1, labels_truncated.shape[1])), columns=labels_truncated.columns.values.tolist())
            labels_truncated = labels_truncated.append(zero_padding, ignore_index=True)
            if X_truncated.shape[0] == nrOfSteps:
                break
        return X_truncated, labels_truncated
    # else return nan
    else:
        return pd.DataFrame(np.nan, index=[0], columns=['A']), pd.DataFrame(np.nan,  index=[0], columns=['A'])


def truncate_all(X_ts, labels, duration, part, discard=True):
    """
    @author: Daniel
    truncates data as well as labels based on a duration [s] and a flag indicating which part of the signal shall be used.
    Three options can be called 'first', 'center', 'last'
    :param X_ts: Pandas Dataframe of time series
    :param labels: labels in a Pandas Dataframe format as from get TimeDistributedLabels()
    :param duration: duration of the truncated signal in seconds
    :param part: string either 'first', 'center' or 'last'
    :param discard: not recommended to set to False as zero padding is not implemented efficiently
    :return: truncated time series in Dataframe format
    """
    X_truncated = pd.DataFrame()
    labels_truncated = pd.DataFrame()
    # iterate over all time sersies
    for stopId in X_ts['stopId'].unique():
        # call function that truncates a single time series and distributed label
        X_single, label_single = truncate_single(X_ts.loc[X_ts['stopId'] == stopId],
                                                 labels.loc[labels['stopId'] == stopId], duration, part, discard)
        # adding truncated time series and label to data frame, if not nan (to short for window)
        if not X_single.isnull().values.any():
            X_truncated = pd.concat([X_truncated, X_single])
            labels_truncated = pd.concat([labels_truncated, label_single])
    return X_truncated, labels_truncated


def truncate_differentiated(X_ts, labels, part, target_list, unique=True):
    discard = True
    same_length_list = list()
    for i in range(len(target_list)):
        target = target_list[i]
        duration = target[0]
        X_truncated = pd.DataFrame()
        labels_truncated = pd.DataFrame()
        # iterate over all time sersies
        for stopId in X_ts['stopId'].unique():
            X_curr = X_ts[X_ts['stopId'] == stopId] # inconcurrent syntax TODO
            label_curr = labels.loc[labels['stopId'] == stopId]
            check = X_curr.get_value(X_curr.last_valid_index(), 'time') >= target[0]
            check2 = X_curr.get_value(X_curr.last_valid_index(), 'time') <= (target[0] + target[1])
            if check & check2:
                # call function that truncates a single time series and distributed label
                X_single, label_single = truncate_single(X_curr, label_curr, target[0], part[i], discard)
                # adding truncated time series and label to data frame, if not nan (to short for window)
                if not X_single.isnull().values.any(): # TODO elif for Zero Padding
                    X_truncated = pd.concat([X_truncated, X_single])
                    labels_truncated = pd.concat([labels_truncated, label_single])
                    if unique:
                        X_ts = X_ts[X_ts['stopId'] != stopId]
                        labels = labels[labels['stopId'] != stopId]
        same_length_list.append((X_truncated, labels_truncated))
    return same_length_list


def reduceLabel(labels_distributed):
    """
    @author: Daniel
    reduces time distributed labels back to a single label for a time series, based on either its stop id or its
    windowed sliceId
    :param labels_distributed: in the output format of getTimeDistributedLabels()
    :return: Pandas Format Labels
    """
    COLUMN_ID = 'stopId'
    if 'sliceId' in labels_distributed.columns: # if windowed signals are used, the id is switched to the sliceId
        COLUMN_ID = 'sliceId'
    index = 0
    labels_single = list()
    # iterate over all unique IDs
    for id in labels_distributed[COLUMN_ID].unique():
        # for each id the maximum of the label function is searched, resulting in either 0 or 1
        maximum = max(labels_distributed.loc[labels_distributed[COLUMN_ID] == id, 'label'])
        labels_single.append(maximum)
    return pd.DataFrame(labels_single)


def shape_Data_to_LSTM_format(X_ts, dropChannels=['stopId', 'time'], scale = True):
    """
    @author: Daniel
    shapes data sets from Pandas Dataformat into numpy array format, needed for training LSTM network.
    with a vector containing strings with column names specific columns can be dropped. at least the stopId and time should
    be dropped as they do not contain relevant information for training
    :param X_ts: Pandas Dataframe format
    :param dropChannels: list of strings
    :return: 3D array of data in the format [samples, timestepts, features]
    """
    COLUMN_ID = 'stopId'
    if 'sliceId' in X_ts.columns:   # if windowed signals are used, the id is switched to the sliceId
        COLUMN_ID = 'sliceId'
        dropChannels.append(COLUMN_ID)  # add sliceId to drop list
    X_numpy = []
    for id in X_ts[COLUMN_ID].unique():
        # drop unnecessary columns and then take values with pandas values attribute
        X = np.array(dropDataChannels(X_ts.loc[X_ts[COLUMN_ID] == id], dropChannels).values)
        # scale Data to [0 1]
        if scale:
            scaler = MinMaxScaler()
            MinMaxScaler(copy=True, feature_range=(0, 1))
            X = scaler.fit_transform(X)

        X_numpy.append(X)
    return np.array(X_numpy)


def shape_Labels_to_LSTM_format(labels):
    """
    @author: Daniel
    shapes data into numpy format, fit for LSTM training. Labels can be inputed as single label, resulting in a
    one dimensional output (needed for a Dense Layer, following a Flatten Layer). If Time-Distributed labels are inputed
    a three-dimensional array is outputed, needed for a TimeDistributed(Dense()) Layer
    :param labels: Pandas data frame formated labels
    :return: labels in numpy format
    """
    COLUMN_ID = 'stopId'
    if 'sliceId' in labels.columns:   # if windowed signals are used, the id is switched to the sliceId
        COLUMN_ID = 'sliceId'

    nrOfLabels = len(labels[COLUMN_ID].unique())
    shape_one = np.array(labels.loc[labels[COLUMN_ID] == labels[COLUMN_ID].unique()[0], 'label']).shape
    label_numpy = np.zeros((nrOfLabels, shape_one[0]))
    # iterate over all unique values
    count = 0
    for id in labels[COLUMN_ID].unique():
        # add labels as read from the dataframe
        label_numpy[count] = np.array(labels.loc[labels[COLUMN_ID] == id, 'label'].values)
        count += 1
    # transform list into numpy array
    v = np.array(label_numpy)

    if v.shape[1] != 1:
        # reshape into 3D array [samples, timestepts, features]
        return v.reshape((v.shape[0], v.shape[1], 1))
    else:
        # otherwise 1D
        return v


def getClassWeight_Dict(labels):
    nrOnes = sum(labels)  # calc number of data sets that squeal
    nrZeros = len(labels) - nrOnes  # calc number of data sets that do not squeal
    class_weight = {0: 1.,
                    1: math.floor(nrZeros/nrOnes)
                    }
    return  class_weight


def getUpDownLabel_pandas(X_ts, labels_td, upTime = 0.1, downTime=0.1, doUp=True, doDown=True, COLUMN_ID = 'stopId'):
    sampleTime = labels_td.at[labels_td.first_valid_index() + 1, 'time'][0] - \
                 labels_td.at[labels_td.first_valid_index(), 'time'][0]
    nrIndexesDown = math.floor(downTime / sampleTime)
    nrIndexesUp = math.floor(upTime / sampleTime)

    X_without = X_ts
    labels_without = labels_td
    X_with = X_ts
    labels_with = labels_td
    for id in labels_td[COLUMN_ID].unique():
        squeal = bool(labels_td[labels_td[COLUMN_ID] == id].max()[0])
        if squeal:
            X_without = X_without[X_without[COLUMN_ID] != id]
            labels_without = labels_without[labels_without[COLUMN_ID] != id]
        elif not squeal:
            X_with = X_with[X_with[COLUMN_ID] != id]
            labels_with = labels_with[labels_with[COLUMN_ID] != id]

    columnsPDLabel = labels_with.columns.values.tolist()
    labels_upDown_pandas = pd.DataFrame(columns=columnsPDLabel)
    for id in labels_with[COLUMN_ID].unique():
        # each label for its own for robustness reasons
        currLabel = labels_with[labels_with[COLUMN_ID] == id]
        indexOne = currLabel.index[currLabel['label'] == 1].tolist()
        for i in range(len(indexOne)):
            if (indexOne[i] > currLabel.first_valid_index()) & doUp:
                if int(currLabel.at[indexOne[i] - 1, 'label']) == 0:
                    for k in range(1, nrIndexesUp + 1):
                        # check range of signal
                        if indexOne[i] - k >= currLabel.first_valid_index():
                            currLabel.at[indexOne[i] - k, 'label'] = 2
            if (indexOne[i] < currLabel.last_valid_index()) & doDown:
                if int(currLabel.at[indexOne[i] + 1, 'label']) == 0:
                    for k in range(1, nrIndexesDown + 1):
                        if indexOne[i] + k <= currLabel.last_valid_index():
                            currLabel.at[indexOne[i] + k, 'label'] = 3
        labels_upDown_pandas = labels_upDown_pandas.append(currLabel)
    return X_without, X_with, labels_without, labels_upDown_pandas


def getUpDownLabels_np(labels_upDown_pandas, COLUMN_ID='stopId'):


    #### one hot encoding: no-squeal y/n, squeal y/n, up y/n, down y/n
    label_np = np.zeros((len(labels_upDown_pandas[COLUMN_ID].unique()),
                         len(labels_upDown_pandas[labels_upDown_pandas[COLUMN_ID] ==
                                                  labels_upDown_pandas[COLUMN_ID].unique()[0]]), 4))
    for i in range(label_np.shape[0]):
        id = labels_upDown_pandas[COLUMN_ID].unique()[i]
        currLabel = labels_upDown_pandas[labels_upDown_pandas[COLUMN_ID] == id]['label'].values
        squealYN_index = np.where(currLabel == 1)[0]
        upYN_index = np.where(currLabel == 2)[0]
        downYN_index = np.where(currLabel == 3)[0]

        for index in squealYN_index:
            label_np[i][index][1] = 1
        for index in upYN_index:
            label_np[i][index][2] = 1
        for index in downYN_index:
            label_np[i][index][3] = 1
    return label_np


def upSample(X_ts, labels_td, SamplingFactor, COLUMN_ID='stopId'):
    X_interpolated = pd.DataFrame()
    labels_interpolated = pd.DataFrame()
    for stopId in X_ts[COLUMN_ID].unique():

        X_curr = X_ts[X_ts[COLUMN_ID] == stopId]
        initialLength = len(X_curr)
        label_curr = labels_td.loc[labels_td[COLUMN_ID] == stopId]

        columndsX = X_curr.columns.values.tolist()
        columnsy = label_curr.columns.values.tolist()
        nanXFrame = pd.DataFrame(np.nan, index=range(SamplingFactor-1), columns=columndsX)
        nanyFrame = pd.DataFrame(np.nan, index=range(SamplingFactor-1), columns=columnsy)
        X_curr_int = pd.DataFrame()
        label_curr_int = pd.DataFrame()
        for index in range(initialLength):
            X_curr_int = X_curr_int.append(X_curr.iloc[index, :], ignore_index=True)
            X_curr_int = X_curr_int.append(nanXFrame, ignore_index=True)
            label_curr_int = label_curr_int.append(label_curr.iloc[index, :], ignore_index=True)
            label_curr_int = label_curr_int.append(nanyFrame, ignore_index=True)



        X_curr_int['dec1'] = X_curr_int['dec1'].interpolate(method='linear')
        X_curr_int['tempg'] = X_curr_int['tempg'].interpolate(method='linear')
        X_curr_int['frc1'] = X_curr_int['frc1'].interpolate(method='linear')
        X_curr_int['n1'] = X_curr_int['n1'].interpolate(method='linear')

        label_curr_int['label'] = label_curr_int['label'].interpolate(method='nearest')
        label_curr_int['time'] = label_curr_int['time'].interpolate(method='linear')
        if COLUMN_ID=='sliceId':
            fillStopId = X_curr['stopId'].unique()[0]
            label_curr_int['stopId'] = label_curr_int['stopId'].fillna(fillStopId)
            X_curr_int['stopId'] = X_curr_int['stopId'].fillna(fillStopId)
        X_curr_int[COLUMN_ID] = X_curr_int[COLUMN_ID].fillna(stopId)
        label_curr_int[COLUMN_ID] = label_curr_int[COLUMN_ID].fillna(stopId)
        X_curr_int = X_curr_int.interpolate(method='spline', order=2)

        X_interpolated = pd.concat([X_interpolated, X_curr_int])
        labels_interpolated = pd.concat([labels_interpolated, label_curr_int])

    return X_interpolated, labels_interpolated


# Log Functions
def summarizeFeatureSelection(X_feat, X_feat_relevant):

    # todo: tsfresh output in separate file?
    numExtractedFeatures = X_feat.shape[1]
    numSelectedFeatures = X_feat_relevant.shape[1]
    noOfExamplesUsedForSelection = X_feat_relevant.shape[0]
    irrelevantFeatures = numExtractedFeatures - numSelectedFeatures
    title = "Summary of Feature Generation: \n"

    summarySelection = "  Selected Features: " + str(numSelectedFeatures) + "/" + str(numExtractedFeatures) +\
                       " (" + str(irrelevantFeatures) + " irrelevant)" + "\n" + \
                       "  Examples used for Selection: " + str(noOfExamplesUsedForSelection)

    listOfFeatures = sorted(getFeatureNames(X_feat_relevant))
    selectedFeatures = '\n'.join('{}: {}'.format(*k) for k in enumerate(listOfFeatures))

    logmsg = "\n".join([title, summarySelection, selectedFeatures, "\n"])

    logging.info(logmsg)

def summarizeDataSplitting(trainData, testData, valData):

    numOfTrainData= len(trainData.X)
    numOfTestData = len(testData.X)
    numOfValData  = len(valData.X)
    numOfTotalExamples = numOfTrainData + numOfTestData + numOfValData

    percentTraining = round(numOfTrainData/numOfTotalExamples * 100)
    percentTest     = round(numOfTestData/numOfTotalExamples * 100)
    percentVal      = round(numOfValData/numOfTotalExamples * 100)

    numOfLabel1InTrainData = len(trainData.labels[trainData.labels == 1])
    numOfLabel1InTestData  = len(testData.labels[testData.labels == 1])
    numOfLabel1InValData   = len(valData.labels[valData.labels == 1])

    percentOfLabel1InTrain = round(numOfLabel1InTrainData/numOfTrainData * 100)
    percentOfLabel1InTest  = round(numOfLabel1InTestData/numOfTestData * 100)
    percentOfLabel1InVal   = round(numOfLabel1InValData/numOfValData * 100)

    numOfLabel0InTrainData = numOfTrainData - numOfLabel1InTrainData
    numOfLabel0InTestData  = numOfTestData - numOfLabel1InTestData
    numOfLabel0InValData   = numOfValData - numOfLabel1InValData

    title = "Summary of Data Splitting: \n"

    relations = "  Percentages for Splitting: " + "\n" + \
                "    Training: "   + str(percentTraining) +  "% (= " + str(numOfTrainData) + " Examples)" + "\n" + \
                "    Test: "       + str(percentTest)     +  "% (= " + str(numOfTestData)  + " Examples)" + "\n" + \
                "    Validation: " + str(percentVal)      +  "% (= " + str(numOfValData)   + " Examples)" + "\n"

    balance = "  Balance after Splitting: " + "\n" + \
              '    Squeals in Trainingset: '    +  str(numOfLabel1InTrainData) + "/" +  str(numOfTrainData) +\
                                              "(= " + str(percentOfLabel1InTrain) + " %) \n" +\
              '    Squeals in Testset: '        + str(numOfLabel1InTestData)  + "/" + str(numOfTestData) +\
                                              "(= " + str(percentOfLabel1InTest)  + " %) \n" + \
              '    Squeals in Validationset: '  + str(numOfLabel1InValData)   + "/" + str(numOfValData) +\
                                              "(= " + str(percentOfLabel1InVal)   + " %) \n"

    logmsg = "\n".join([title, relations, balance])

    logging.info(logmsg)

def summarizeBalancing(labels, labels_balanced):
    numOfNoSquealsOriginal     = len(labels[labels == 0])
    numOfSquealsOriginal       = len(labels[labels == 1])
    numOfTotalExamplesOriginal = len(labels)

    numOfNoSquealsBalanced     = len(labels_balanced[labels_balanced==0])
    numOfSquealsBalanced       = len(labels_balanced[labels_balanced==1])
    numOfTotalExamplesBalanced = len(labels_balanced)

    percentOfSquealsOriginal   = round(numOfSquealsOriginal/numOfTotalExamplesOriginal *100)
    percentOfSquealsBalanced   = round(numOfSquealsBalanced/numOfTotalExamplesBalanced * 100)
    percentOfNoSquealsBalanced = 100-percentOfSquealsBalanced
    percentOfNoSquealsOriginal = 100-percentOfSquealsOriginal

    numOfRemovedExamples = numOfTotalExamplesOriginal - numOfTotalExamplesBalanced

    if numOfTotalExamplesOriginal == numOfTotalExamplesBalanced:
        title = 'Keep Original Balance' + "\n"

    else:
        title = 'Change Balance of Data' + "\n"

    msg = title + "\n" + \
         "  Balance before Splitting:" + "\n" +\
         '    Squeals: '   + str(numOfSquealsBalanced)   + '/' + str(numOfTotalExamplesBalanced) +\
         " (" + str(percentOfSquealsBalanced) + '%)' + "\n" +\
         '    No Squeals: ' + str(numOfNoSquealsBalanced) + '/' + str(numOfTotalExamplesBalanced)  +\
         " (" + str(percentOfNoSquealsBalanced) + '%)' + "\n" +\
         "    Examples left: " + str(numOfTotalExamplesBalanced) +\
         ' (' + str(numOfRemovedExamples)+ "/" + str(numOfTotalExamplesOriginal) + " removed)" +"\n"

    logging.info(msg)
