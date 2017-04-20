import random
import nltk
#https://www.strehle.de/tim/weblog/archives/2015/09/03/1569
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

with open('./test_data/moon.txt') as t:
  text = t.read().lower()
  
words = nltk.Text(text)   
fdist = nltk.FreqDist(words)

for word, frequency in fdist.most_common(25):
  print(u'{};{}'.format(word, frequency))