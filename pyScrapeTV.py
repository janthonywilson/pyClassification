'''
Created on Aug 31, 2015

@author: Tony
'''
import time
import requests
from lxml import html, etree
from bs4 import BeautifulSoup
# from BeautifulSoup import BeautifulSoup

sTime = time.time()

# url = "http://www.tv.com/#{tag}/show/#{id}/episode_guide.html?season=0&tag=season_dropdown%3Bdropdown"
url = 'http://www.tv.com/shows/game-of-thrones/mothers-mercy-3079913/'
# url = 'http://www.tv.com/shows/game-of-thrones/mothers-mercy-3079913/usersubmission/episode_synopsis.html'
# url ='http://www.tv.com/shows/game-of-thrones/mothers-mercy-3079913/usersubmission/episode_synopsis.html/usersubmission/episode_synopsis.html?show_id=77121&amp;episode_id=3079913/'
page = requests.get(url)
# tree = html.fromstring(page.text)
bs = BeautifulSoup(page.text, 'html.parser')
# print(bs.prettify())

# des = bs.findAll('div', { 'class' : 'description' })
des = bs.findAll('div', { 'itemprop' : 'description' })

print des
# print tree.TextareaElement
# <div class="description" itemprop="description">

# find_text = etree.XPath('description')
# text = find_text(tree)[0]




print("Duration: %s" % (time.time() - sTime))
