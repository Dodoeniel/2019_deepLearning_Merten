"""Module for Configuration of log process"""

import logging
import time
import os

def configureLogfile(logpath, logname):

    if not os.path.isdir(logpath):
        os.makedirs(logpath)

    logfileName = logpath + timeAsString() + "_" + logname + "_" + ".log"

    logging.basicConfig(
        filename=logfileName,
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Capture all warnings into the logfile(e.g. from tsfresh)
    logging.captureWarnings(True)

def timeAsString():
    timestr = time.strftime("%Y-%m-%d_%Hh%M_%S")
    return timestr

def writeLogfileHeader(config):
    """ Header for Logfile"""

    attrs = vars(config)
    header = "Project Configuration: \n" + "\n".join("  %s: %s" % item for item in attrs.items()) + "\n"
    logging.info(header)

def resetLogConfigurations():
    loggers = logging.getLogger()
    for handler in list(loggers.handlers):
        loggers.removeHandler(handler)