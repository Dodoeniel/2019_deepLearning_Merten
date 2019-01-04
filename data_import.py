"""library for importing brake data in a safe way

@author: Nadine
"""

import math
import pandas as pd
import logging

from Libraries import prepare_csv as csv

# Import Functions

def loadVerifiedBrakeData(eedPath, eecPath, datasetNumber):
    """Loads Data and corresponding Labels from 'Brake Squeal' Data Set.

       Only considers Data for which a label is found

       Returns:
            X_ts: Flat DataFrame
                Contains all time series Data from eed-Files
                several rows per stopId, therefore not used as an index
                Example:
                (index)  stopId  time quantity1 quantity2
                         --------------------------------
                  0      1.1501   1     ...       ...
                  1      1.1501   2     ...       ..
                  2      1.1501   3     ...       ...
                  3      2.1501   1     ...       ...
                  4      2.1501   2     ...       ...

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

       """

    eedData = csv.eed_csv_to_eedData(eedPath, datasetNumber)
    eecData = csv.eec_csv_to_eecData(eecPath, datasetNumber)

    X_ts, labels = performDataVerification(eedData, eecData)

    summarizeImportResult(eedData, eecData, X_ts, labels)

    return X_ts, labels

def performDataVerification(eedData, eecData):
    """find for every eed-File a matching entry, i.e. label in eec-File

       data for which no label is found is deleted

       eedData: FlatDf
       eecData: Df

       return:
         # X_ts: flatDf, labels: Series
       """

    matchedEedTimeSeries   = []

    labelValues = []
    labelStopIds = []

    for stopId, eedDataForStopId in eedData.groupby('stopId'):

        matchingEntry = findMatchingEntryInEECData(eecData, eedDataForStopId)

        if matchingEntry is not None:
            label = getLabelFromEntry(matchingEntry)
            labelValues.append(label)
            labelStopIds.append(stopId)
            matchedEedTimeSeries.append(eedDataForStopId)
        else:
            msg = 'No matching Entry was found for StopId: ' + stopId + '. Exclude StopId from Dataset.' + "\n"
            logging.warning(msg)

    X_ts = pd.concat(matchedEedTimeSeries)
    y = pd.Series(data=labelValues, index=labelStopIds)

    return X_ts, y

def findMatchingEntryInEECData(eecData, eedDataForStopId):
    matchingEntry = None

    stopId = eedDataForStopId['stopId'].unique()[0]
    eecDataForStopId = eecData[eecData['stopId'] == stopId]

    if len(eecDataForStopId) > 1:
        logging.info("Multiple entries with stopid=" + stopId + " in eec file. Searching for first matching entry.")

    if len(eecDataForStopId) == 0:
        logging.info("No entry with stopId=" + stopId + " in eec file.")

    for index, eecEntry in eecDataForStopId.iterrows():

        if isMatching(eecEntry, eedDataForStopId):
            matchingEntry = eecEntry
            break

    return matchingEntry

def isMatching(eecEntry, eedDataForStopId):
    """Checks, if data from eec file fits to data from eed file, double checking the assignment via stopIds

    Format of X: (from eed)
        stopId time trg1 n1 v1 p1
        1.1051 0.01
        1.1051 0.02

    Format of CheckData: (from eec)
        stopId stop sect p1max.. stopname
        1.1051
        2.1051
    """

    # larger threshold for ambient temperature due to absolute offest of 0.1 in about 100 examples from dataset 1051
    # deviation is assumed to be caused by different measuremet systems
    tolerance = 0.001
    toleranceForAmbientTemperature = 0.01
    toleranceForMeantChecks = 0.03

    checks = [
        (checkMaxValue, 'p1_max', 'p1', tolerance),
        (checkMaxValue, 'torq1_max', 'torq1', tolerance),
        (checkMaxValue, 'frc1_max', 'frc1', tolerance),

        (checkMinValue, 'p1_min', 'p1', tolerance),

        (checkStartValue, 'p1_start', 'p1', tolerance),
        (checkStartValue, 'trot1_start', 'trot1', tolerance),
        (checkStartValue, 'tfld1_start', 'tfld1', tolerance),
        (checkStartValue, 'v1_start', 'v1', tolerance),
        (checkStartValue, 'tamb1_start', 'tamb1', toleranceForAmbientTemperature),

        (checkEndValue, 'p1_end', 'p1', tolerance),
        (checkEndValue, 'trot1_end', 'trot1', tolerance),
        (checkEndValue, 'tfld1_end', 'tfld1', tolerance),
        (checkEndValue, 'stoptime', 'time', tolerance),
        (checkEndValue, 'v1_end', 'v1', tolerance),

        (checkMeantValue,'p1_meant', 'p1', toleranceForMeantChecks),
        (checkMeantValue, 'torq1_meant', 'torq1', toleranceForMeantChecks),
        (checkMeantValue, 'frc1_meant', 'frc1', toleranceForMeantChecks)
    ]

    matchingResult = True

    for checkFunction, eecColumnName, eedColumnName, tolerance in checks:
        checkPassed = checkFunction(eecEntry, eedDataForStopId, eecColumnName, eedColumnName, tolerance)

        if not checkPassed:
            matchingResult = False

    return matchingResult

def getLabelFromEntry(entryOfEec):
    if math.isnan(entryOfEec['f_1']):
        label = 0
    else:
        label = 1
    return label


## Get Functions
def checkMaxValue(eecEntry, eedDataForStopId, eecColumn, eedColumn, tolerance):

    max_eec = eecEntry[eecColumn]
    max_eed = eedDataForStopId[eedColumn].max()

    checkPassed = floatsAreEqual(max_eec, max_eed, tolerance)

    if not checkPassed:
        stopId = eedDataForStopId['stopId'].unique()[0]
        reportFailureOfCheck('checkMaxValue', eecColumn, max_eec, eedColumn, max_eed, stopId)

    return checkPassed

def checkMinValue(eecEntry, eedDataForStopId, eecColumn, eedColumn, tolerance):

    min_eec = eecEntry[eecColumn]
    min_eed = eedDataForStopId[eedColumn].min()

    checkPassed = floatsAreEqual(min_eec, min_eed, tolerance)

    if not checkPassed:
        stopId = eedDataForStopId['stopId'].unique()[0]
        reportFailureOfCheck('checkMinValue', eecColumn, min_eec, eedColumn, min_eed, stopId)

    return checkPassed

def checkStartValue(eecEntry, eedDataForStopId, eecColumn, eedColumn, tolerance):

    starttime = eedDataForStopId['time'].min()

    start_eec = eecEntry[eecColumn]
    start_eed = eedDataForSpecificTime(starttime, eedColumn, eedDataForStopId)

    checkPassed = floatsAreEqual(start_eec, start_eed, tolerance)

    if not checkPassed:
        stopId = eedDataForStopId['stopId'].unique()[0]
        reportFailureOfCheck('checkStartValue', eecColumn, start_eec, eedColumn, start_eed, stopId)

    return checkPassed

def checkEndValue(eecEntry, eedDataForStopId, eecColumn, eedColumn, tolerance):

    endtime = eedDataForStopId['time'].max()

    end_eec = eecEntry[eecColumn]
    end_eed = eedDataForSpecificTime(endtime, eedColumn, eedDataForStopId)

    checkPassed = floatsAreEqual(end_eec, end_eed, tolerance)

    if not checkPassed:
        stopId = eedDataForStopId['stopId'].unique()[0]
        reportFailureOfCheck('checkEndValue', eecColumn, end_eec, eedColumn, end_eed, stopId)

    return checkPassed

def checkMeantValue(eecEntry, eedDataForStopId, eecColumn, eedColumn, tolerance):

    meant_eec = eecEntry[eecColumn]
    meant_eed = calculateMeant(eedDataForStopId, eedColumn)

    checkPassed = floatsAreEqual(meant_eec, meant_eed, tolerance)

    if not checkPassed:
        stopId = eedDataForStopId['stopId'].unique()[0]
        reportFailureOfCheck('checkMeantValue', eecColumn, meant_eec, eedColumn, meant_eed, stopId)

    return checkPassed


# Helper Functions
def calculateMeant(eedDataForStopId, eedColumn):

    duration = eedDataForStopId['time'].max()
    samplingPeriod = getSamplingPeriod(eedDataForStopId)

    weightedSamples = eedDataForStopId[eedColumn].multiply(samplingPeriod)
    meant = weightedSamples.sum()/duration

    return meant

def getSamplingPeriod(eedDataForStopId):
    """assumes uniformly sampled signal"""
    samplingPeriod = eedDataForStopId['time'].max()/len(eedDataForStopId)
    return samplingPeriod

def eedDataForSpecificTime(timepoint, eedColumn, eedDataForStopId):

    row = eedDataForStopId[eedDataForStopId['time'] == timepoint]
    if len(row) is not 1:
        raise IndexError("Timepoint time=" + str(timepoint) + " is not unique in: \n" + str(row))

    eedData = row[eedColumn].iloc[0]
    return eedData

def floatsAreEqual(floatA, floatB, tolerance = 0.001):

    if floatA >= 0:
        upperLimit = floatA + floatA * tolerance
        lowerLimit = floatA - floatA * tolerance
    else:
        upperLimit = floatA - floatA * tolerance
        lowerLimit = floatA + floatA * tolerance

    if lowerLimit <= floatB <= upperLimit:
        isEqual = True
    else:
        isEqual = False

    return isEqual


# Log Function
def summarizeImportResult(eedData, eecData, X_ts, labels):

    numEedFiles = len(eedData['stopId'].unique())
    numUsedEedFiles = len(X_ts['stopId'].unique())
    numUnusedEedFiles = numEedFiles - numUsedEedFiles

    numEntriesInEecFile = len(eecData)
    numUsedEntriesInEecFile = len(labels)
    numUnusedEntriesInEecFile = numEntriesInEecFile - numUsedEntriesInEecFile

    title = "Summary of Data Import:"

    summaryEED = "Eed-files used: " + str(numUsedEedFiles) + "/" + str(numEedFiles) + \
                  " (" + str(numUnusedEedFiles) + " unused)"

    summaryEEC = "Eec-entries used: " + str(numUsedEntriesInEecFile) + "/" + str(numEntriesInEecFile) + \
                  " (" + str(numUnusedEntriesInEecFile)+ " unused)"

    logmsg = "\n".join([title, summaryEED, summaryEEC])

    logging.info(logmsg)

def reportFailureOfCheck(functionName, eecColumnName, eecValue, eedColumnName, eedValue, stopId):

    failureMsg = 'StopId '+ stopId +   ': Failure of ' + functionName + ' for: ' \
                 + eecColumnName + '=' + str(eecValue) + ' and ' + eedColumnName + '=' + str(eedValue)

    logging.warning(failureMsg)
