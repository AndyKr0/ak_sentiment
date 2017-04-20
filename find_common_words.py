# Code adapted from: 
# https://www.strehle.de/tim/weblog/archives/2015/09/03/1569

import sys
import os
import csv
import codecs

import nltk
from nltk.corpus import stopwords


# NLTK's default English stopwords
default_stopwords = set(nltk.corpus.stopwords.words('english'))

# We're adding some on our own - could be done inline like this...
# custom_stopwords = set((u'-', u'dass', u'mehr'))
# ... but let's read them from a file instead (one stopword per line, UTF-8)
stopwords_file = './stopwords.txt'

custom_stopwords = set(codecs.open(stopwords_file, 'r', 'utf-8').read().splitlines())

all_stopwords = default_stopwords | custom_stopwords






def getFrequentWords(input, size, speaker_name, year):
 
  words = nltk.word_tokenize(input.read())

  # Remove single-character tokens (mostly punctuation)
  words = [word for word in words if len(word) > 1]

  # Remove numbers
  words = [word for word in words if not word.isnumeric()]

  # Lowercase all words (default_stopwords are lowercase too)
  words = [word.lower() for word in words]

  # Stemming words seems to make matters worse, disabled
  # stemmer = nltk.stem.snowball.SnowballStemmer('german')
  # words = [stemmer.stem(word) for word in words]

  # Remove stopwords
  words = [word for word in words if word not in all_stopwords]

  # Calculate frequency distribution
  fdist = nltk.FreqDist(words)

  # # Output top 20 words
  # for word, frequency in fdist.most_common(20):
  #     print(u'{}\t{}'.format(frequency, word))

  word_array = []
  word_array.append(year)
  word_array.append(speaker_name)

  for word, frequency in fdist.most_common(size):
    word_array.append(str(word))

  return word_array



source_dir = '/home/cabox/workspace/inagural'
files_to_analyze = os.listdir(source_dir)

for file in files_to_analyze:
  speaker_name = file.split('.')[0].split('-')[1]
  year = file.split('.')[0].split('-')[0]
  #print("Analyzing " + file)

  # Open text file
  with codecs.open(source_dir + '/' + file, 'r', 'utf-8') as input_file:
      print ','.join(getFrequentWords(input_file, 20, speaker_name, year))

     
