'''
Created on Aug 31, 2015

@author: Tony
'''
import os
import csv
import time
import json
import urllib2
from bs4 import BeautifulSoup
# import requests


sTime = time.time()
MAXPAGE = 18
with open('data' +os.sep+ 'tmp.txt', 'wb') as outputFileObj:
    outputFileObj.write('ID\tShow Name\tEpisode Name\tGenre\tRating\tSummary\n')

    page = 1
    while page < MAXPAGE:
        url = 'http://api.tvmaze.com/shows?page=' + str(page)
        data = urllib2.urlopen(url).read()
        
    #     result = json.loads(data)
        result = json.loads(data.decode('utf-8'))
        
        for i in xrange(0, len(result)):
            
            genre = ','.join(result[i]['genres'])
            genre = genre.encode('utf-8').strip()
                
            name = result[i]['name']
            name = name.encode('utf-8').strip()
            
            rating = result[i]['rating']['average']
            if rating is None:
                rating = 'none'
            elif isinstance(rating, float):
                rating = str(rating)
            elif isinstance(rating, int):
                rating = str(float("{0:.1f}".format(float(rating))))
#                     rating = rating.encode('utf-8').strip()
            
            summary = BeautifulSoup(result[i]['summary'], 'lxml').text
            summary = summary.encode('utf-8').strip()
            summary = summary.replace('\n', '')
                
            showID = result[i]['id']
            episodeURL = 'http://api.tvmaze.com/shows/' +str(showID)+ '/episodes'
            episodeData = urllib2.urlopen(episodeURL).read()
            episodeResult = json.loads(episodeData.decode('utf-8'))
            
            
            if genre and summary:
#                 genre = genre.split(',')
                outputFileObj.write(str(showID) +'\t'+ name +'\t'+ name +'\t'+ genre +'\t'+ rating +'\t'+ summary + '\n')
                
    #             print result[i]['name'], result[i]['genres'], BeautifulSoup(result[i]['summary'], 'html.parser').text
                
                for j in xrange(0, len(episodeResult)):
                    if episodeResult[j]['summary']:
#                         print episodeResult[j]
                        episodeName = episodeResult[j]['name']
                        episodeName = episodeName.encode('utf-8').strip()
                
                        episodeSummary = BeautifulSoup(episodeResult[j]['summary'], 'lxml').text
                        episodeSummary = episodeSummary.encode('utf-8').strip()
                        episodeSummary = episodeSummary.replace('\n', '')
                        
                        outputFileObj.write(str(showID) +'\t'+ name +'\t'+ episodeName +'\t'+ genre +'\t'+ rating +'\t'+ episodeSummary + '\n')
            
        page += 1
        print page








# showSearchURL = 'http://api.tvmaze.com/search/shows?q=%s'
# showInfoURL =  'http://api.tvmaze.com/shows/%s'
# scheduleURL = 'http://api.tvmaze.com/schedule?country=US&date=%s'


print('Duration: %s' % (time.time() - sTime))