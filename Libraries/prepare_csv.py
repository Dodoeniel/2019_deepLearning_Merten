"""library for import of csv Brake Data

@authors: nadine, quy """

import glob
import os
import pandas as pd

# Default values for brake data
DELIMITER = ";"
LINEBREAK = "\n"
DATA_BEGIN = "brake_data_begin"
DATA_END = "brake_data_end"
FIRST_DATA_ROW = 3
FILE_ENDING_EED = '*eed.csv'
FILE_ENDING_EEC = '*eec.csv'
COL_HEAD_ID_STRING = 'brake_identifier'
HEADER_BRAKE_NO_TAG = 'current_no_of_cycle' # Name of current id in header



def eec_csv_to_eecData(eecPath, datasetNumber):
    """ reads data from eec.csv-file into a DataFrame format

    @author: Nadine

    Parameters:
        eecPath: String
            path to eec-file including filename and fileending
        datasetNumber: String
            number of dataset.

    Returns: eecData: DataFrame
            contains Data From eec File

            stopId  max(quantity1) start(quantity2)
            ---------------------------------------
            1.1501     ...             ...
            2.1501     ...             ...
            3.1501     ...             ...
            4.1501     ...             ...
            5.1501     ...             ...
    """

    #Check if .eec-file exists
    if not fileExists(eecPath):
        raise FileNotFoundError(eecPath)

    # Read Data from csv to Data Frame
    [header, data] = read_header_data(eecPath)
    del header
    data = replace_decimal_sep(data)
    (col_names, data) = simplify_col_head(data, FIRST_DATA_ROW)
    eecData = pd.DataFrame(data, columns=col_names)

    # Create new StopIds including datasetNumber
    stop_keys = []
    for element in eecData['stop']:
        stop_keys.append(str(element) + '.' + datasetNumber)
    df_stop_keys = pd.DataFrame(stop_keys)
    df_stop_keys.columns = ['stopId']

    # Add StopId Column to dataframe and remove stop column
    eecData = pd.concat([df_stop_keys, eecData], axis=1)
    eecData = eecData.drop(['stop'], axis=1)

    # Convert to float
    excludeColumns = ['stopId']
    eecData = convertStringColumnsToFloat(eecData, excludeColumns)
    return eecData

def eed_csv_to_eedData(eedPath, datasetNumber):
    """ summarizes data from all eed.csv-files into a flat DataFrame format

    @author: Nadine

    Parameters:
        eedPath: String
            path to folder containing eed.csv-files
        datasetNumber: String
            number of dataset. The number is used for creating unique ids

    Returns:
        eedData: Flat DataFrame (tsFresh Format)
            One column per measured quantity. Additionally first columns specifies brakeid and time.
            For each brake action (brake tests) there exist as many rows as time steps.
            Example with datasetNumber 1501:

            stopId  time quantity1 quantity2
            --------------------------------
            1.1501   1     ...       ...
            1.1501   2     ...       ...
            1.1501   3     ...       ...
            2.1501   1     ...       ...
            2.1501   2     ...       ...
    """

    eedFiles = findEedFiles(eedPath)

    eedDataFrames = []
    for eedFile in eedFiles:
        eedDataFrame = readEedFile(eedFile, datasetNumber)
        eedDataFrames.append(eedDataFrame)

    flatDf = pd.concat(eedDataFrames)

    excludeColumns = ['stopId']
    eedData = convertStringColumnsToFloat(flatDf, excludeColumns)

    return eedData

def findEedFiles(eedPath):
    """finds all eed files in given directory

    @author: Nadine

    Parameters:
        eedPath: String
            path to folder containing eed.csv-files

    Returns:
        eedFiles: List of Strings
            contains Names of eedFiles
    """
    eedFiles = glob.glob(eedPath + FILE_ENDING_EED)

    if len(eedFiles) == 0:
        raise FileNotFoundError(eedPath + FILE_ENDING_EED)

    return eedFiles

def readEedFile(eedFile, datasetNumber):
    """reads Data from single eedFile to a single DataFrame

    @author: Nadine

    Parameters:
        eedFile: String
            name of eed.csv-file
        datasetNumber: String
            number of dataset. The number is used for creating unique ids

    Returns: eedFileData: DataFrame

            Example with datasetNumber 1501 and stop = 1:

            stopId  time quantity1 quantity2
            --------------------------------
            1.1501   1     ...       ...
            1.1501   2     ...       ...
            1.1501   3     ...       ...
            1.1501   4     ...       ...
    """

    # Read Data to Data Frame
    [header, data] = read_header_data(eedFile)
    data = replace_decimal_sep(data)
    (col_names, data) = simplify_col_head(data, FIRST_DATA_ROW)

    # add stop id to col_names
    col_names_new = ['stopId']
    col_names_new.extend(col_names)

    # create stop id
    stopId = str(get_current_brake_no(header)) + '.' + datasetNumber

    # Create Data with additional stop id column
    new_data = []
    for i in range(len(data)):
        new_entry = [stopId]
        new_entry.extend(data[i])
        new_data.append(new_entry)

    # Create Data Frame from data with additional stop id
    eedFileData = pd.DataFrame(new_data, columns=col_names_new)

    # Remove empty column heads
    if '' in eedFileData.columns:
        eedFileData.drop([''], axis=1, inplace=True)

    return eedFileData

def convertStringColumnsToFloat(dataFrame, excludeColumns):
    """convert df with string entries to float columns

    columns which can't be converted are excluded
    columns which can be converted but should not, are summarized in excludeColumns

    @author: Nadine
    """

    columnNames = dataFrame.columns.tolist()
    resultColumns = []

    for columnName in columnNames:
        if columnName in excludeColumns:
            unconvertedColumn = dataFrame[columnName]
            resultColumns.append(unconvertedColumn)
        else:
            convertedColumn = pd.to_numeric(dataFrame[columnName], errors="ignore")
            resultColumns.append(convertedColumn)

    convertedDataFrame = pd.concat(resultColumns, axis = 1)

    return convertedDataFrame

def fileExists(pathAndFile):
    """ returns if file Exists

    @author: Nadine
    """
    return os.path.isfile(pathAndFile)



def get_current_brake_no(header_data):
    """
    @author: quy
    Returns current no of cycle (brake application) stored in the header of eed file.
    Return -1 when failed.
    :param list header_data: list of lines which is a list of strings.
    """
    current_id = -1
    for line in header_data:
        if line[0].rstrip() == HEADER_BRAKE_NO_TAG:
            try:
                current_id = int(line[1])
            except ValueError:
                print('Warning: Failed to find current no of cycle. ID is set to -1.')
                return current_id
    return current_id

def read_header_data(filename):
    """
    @author: quy
    Read .csv file and returns header and data separately as a list of rows.
    Using: 
    DATA_BEGIN = "brake_data_begin"
    DATA_END = "brake_data_end"
    """
    file = open(filename,'r', encoding = 'ISO-8859-1')
    print('filename: ', filename)
    lines = [] # list of rows of cells -> [ [x, x, ...], [x, x, ...], ...]
    header = [] # list of rows of cells -> [ [x, x, ...], [x, x, ...], ...]
    data = [] # list of rows of cells -> [ [x, x, ...], [x, x, ...], ...]
    
    for line in file:
        lines.append(line.rstrip(LINEBREAK).split(sep = DELIMITER))
    file.close()
    
    # split header and data for seperated access (e.g. export)
    i = 0
    while lines[i][0] != DATA_BEGIN and i < len(lines):
        header.append(lines[i])
        i += 1
    i += 1
    while lines[i][0] != DATA_END and i < len(lines):
        data.append(lines[i])
        i += 1

    return [header,data]

def simplify_col_head(data, first_data_row):
    """
    @author: quy
    Removes first rows of data and returns column names and rest of data.
    (Dropping unit and description.)
    """
    for col in range(len(data[0])):
        name_temp = data[0][col].replace('@','')
        name_temp = name_temp.replace('&', '_')
        data[0][col] = name_temp
    col_names =  data[0]
    
    del data[0:first_data_row]
    
    return [col_names, data]

def replace_decimal_sep(str_2Dlist):
    """
    @author: quy
    Replaces all ',' to '.' in the 2D list
    """

    for i in range(len(str_2Dlist)):
        for j in range(len(str_2Dlist[i])):
            if isinstance(str_2Dlist[i][j], str):
                str_2Dlist[i][j] = str_2Dlist[i][j].replace(',', '.')
                # str_2Dlist[i][j] = float(str_2Dlist[i][j])
    return str_2Dlist