'''
Created on Oct 21, 2015

@author: Tony
'''

import time
from pySummaryClassify import pySummaryClassify
#===============================================================================

sTime = time.time()

pyClassification = pySummaryClassify()
pyClassification.DataFile = 'SummaryEpisode.txt'
pyClassification.LoadDataFrameObject()
pyClassification.StackPreproc()
pyClassification.CountClasses()
# print pyClassification.ClassesValuesUnique, pyClassification.ClassesValuesCount 
pyClassification.GenFeatures()
pyClassification.ClusterFeatures()
pyClassification.Vis(pyClassification.Features)














print("Duration: %s" % (time.time() - sTime))