'''
Created on Aug 26, 2015

@author: Tony
'''
import os
import csv
import json
import glob
import time
import nltk
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer


sTime = time.time()

CsvList = glob.glob('data' +os.sep+ '*.csv')
DataMap = {}

for inputFiles in CsvList:
    inputNameNoExtension, inputNameExtension = os.path.splitext(inputFiles)
    
    DataFrameObject = pandas.read_csv(inputFiles, header=0, na_values='.')
    DataMap[os.path.basename(inputNameNoExtension)] = DataFrameObject

    
dfUserProfiles = DataMap['UserProfiles_Basic_081515']
dfPanelResponses = DataMap['panel_responses']
dfShowData = DataMap['ShowData']
dfEpisodeData = DataMap['EpisodeData']



    
dfProfilesResponses = pandas.merge(dfUserProfiles, dfPanelResponses, on='user_id')
print dfProfilesResponses.columns.values
dfProfilesResponses.to_csv('data' +os.sep+ 'ProfilesResponses.csv')

dfProfilesResponsesShowEpisode = pandas.merge(dfShowData, dfEpisodeData, on='video_id')
print dfProfilesResponsesShowEpisode.columns.values



print("Duration: %s" % (time.time() - sTime))








#     csvfile = open(inputNameNoExtension + inputNameExtension, 'rb')
#     jsonfile = open(inputNameNoExtension + '.json', 'wb')
#     
#     reader = csv.reader(csvfile)
#     headers = reader.next()
#     reader = csv.DictReader( csvfile, headers )
#     for row in reader:
#         json.dump(row, jsonfile)
#         jsonfile.write('\n')
