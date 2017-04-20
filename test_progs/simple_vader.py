from nltk import tokenize
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

text_to_analyze = open('../test_data/1981-Reagan.txt', 'r')

def vader_instance(text):
  vader_analyzer = SentimentIntensityAnalyzer()
  vader_scores = vader_analyzer.polarity_scores(text)
  return vader_scores

# tokenize sentences
sentence_list = (sent_tokenize(text_to_analyze.read()))

for sentence in sentence_list:
  vader_output = vader_instance(sentence)
  print(sentence)
  print(vader_output)
  print('-----------')
