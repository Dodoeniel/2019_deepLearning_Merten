"""Class for structured Summary of Setup Settings"""

class Configuration:
    def __init__(self, eedPath=None, eecPath=None, datasetNumber=None, projectName =None, savePath=None, X_featPathPickle=None, labelsPathPickle=None):
        self.logTitle = 'Analysis of Brake Squeal Noise with feature based Neural Networks'
        self.logName = 'Logfile_NN'

        self.datasetNumber = datasetNumber  # Used for extended Naming of StopIds
        self.projectName   = projectName    # Used as superior folder name
        self.eedPath = eedPath              # Path (without file name) where eed-Files are stored
        self.eecPath = eecPath              # Path (including filename) where eec-File is stored
        self.savePath = savePath

        # Folder Structure per Dataset
        self.projectPath            = self.projectName
        self.basePath               = self.projectPath + "/Dataset_"  + self.datasetNumber
        self.picklePath             = self.basePath    + "/picklefiles/"
        self.logPath                = self.basePath    + "/logfiles/"

        # Folder Structure per Trainings Run
        # todo: structure run

# Definition of different Configurations
def getConfig_SMP1051(projectName):
    config_SMP1051 = Configuration(eedPath = '/media/computations/DATA/ExperimentalData/SMP_1051/',
                                   eecPath = '/media/computations/DATA/ExperimentalData/SMP_1051/1051_fr_eec.csv',
                                   savePath = '/home/daniel/Documents/MasterArbeitProgrammieren/',
                                   datasetNumber = '1051',
                                   projectName = projectName)
    return config_SMP1051

def getConfig_vereinfacht(projectName):
    config_vereinfacht = Configuration(eedPath = '/media/computations/DATA/ExperimentalData/Vereinfacht/',
                                eecPath = '/media/computations/DATA/ExperimentalData/Vereinfacht/1051_fr_eec.csv',
                                savePath='/home/computations/ExperimentalData/',
                                datasetNumber = '1051',
                                projectName=projectName)
    return  config_vereinfacht

def getConfig_1093(projectName):
    config_1093 = Configuration(eedPath= '../../_Data/1093/eed/',
                                eecPath= '../../_Data/1093/1093_offlinedetection_detection1_validation3_fl_eec.csv',
                                datasetNumber= '1093',
                                projectName=projectName)
    return config_1093

def getConfig_1114(projectName):
    config_1114 = Configuration(eedPath= '../../_Data/1114/eed/',
                                eecPath= '../../_Data/1114/1114_offlinedetection_detection1_validation3_fr_eec.csv',
                                datasetNumber='1114',
                                projectName=projectName)
    return config_1114

def getConfig_1131(projectName):
    config_1131 = Configuration(eedPath= '../../_Data/1131/eed/',
                                eecPath= '../../_Data/1131/1131_offlinedetection_detection1_validation3_fl_eec.csv',
                                datasetNumber='1131',
                                projectName=projectName)
    return config_1131

def getConfig(projectName, callDataset):
    """for parallel execution on cluster only"""
    myDict ={
        #'1051': getConfig_SMP1051,
        '1051': getConfig_vereinfacht,
        '1093': getConfig_1093,
        '1114': getConfig_1114,
        '1131': getConfig_1131,
            }

    config = myDict[callDataset](projectName)

    return  config