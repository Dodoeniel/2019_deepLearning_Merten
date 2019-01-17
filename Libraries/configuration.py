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

def getConfig_systemA(projectName):
    config_systemA = Configuration(eedPath= '/media/computations/DATA/ExperimentalData/systemA/eed/',
                                eecPath= '/media/computations/DATA/ExperimentalData/systemA/1085_fl_eec.csv',
                                savePath='/home/computations/ExperimentalData/',
                                datasetNumber= '1085',
                                projectName=projectName)
    return config_systemA

def getConfig_systemB(projectName):
    config_systemB = Configuration(eedPath= '/media/computations/DATA/ExperimentalData/systemB/eed/',
                                eecPath= '/media/computations/DATA/ExperimentalData/systemB/1093_fl_eec.csv',
                                savePath='/home/computations/ExperimentalData/',
                                datasetNumber= '1093',
                                projectName=projectName)
    return config_systemB


def getConfig_systemC(projectName):
    config_systemC = Configuration(eedPath= '/media/computations/DATA/ExperimentalData/systemC/eed/',
                                eecPath= '/media/computations/DATA/ExperimentalData/systemC/1114_fr_eec.csv',
                                savePath='/home/computations/ExperimentalData/',
                                datasetNumber= '1141',
                                projectName=projectName)
    return config_systemC


def getConfig_systemD(projectName):
    config_systemD = Configuration(eedPath= '/media/computations/DATA/ExperimentalData/systemC/eed/',
                                eecPath= '/media/computations/DATA/ExperimentalData/systemC/1131_fl_eec.csv',
                                savePath='/home/computations/ExperimentalData/',
                                datasetNumber= '1131',
                                projectName=projectName)
    return config_systemD


def getConfig(projectName, callDataset):
    """for parallel execution on cluster only"""
    myDict ={
        '1051': getConfig_SMP1051,
        #'1051': getConfig_vereinfacht,
        'sysA': getConfig_systemA,
        'sysB': getConfig_systemB,
        'sysC': getConfig_systemC,
        'sysD': getConfig_systemD
            }

    config = myDict[callDataset](projectName)

    return  config