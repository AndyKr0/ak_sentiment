# http://stackoverflow.com/questions/34190860/sentiment-analysis-for-sentences-positive-negative-and-neutral
# http://textblob.readthedocs.io/en/dev/advanced_usage.html
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


# tokenize text file into sentences
with open('./test_data/my_sentences.txt', 'r') as infile:
    EXAMPLE_TEXT = infile.read()

blob_default = TextBlob(EXAMPLE_TEXT)
blob_naive = TextBlob(EXAMPLE_TEXT, analyzer=NaiveBayesAnalyzer())

for sentence in blob_default.sentences:
    print(sentence)
    print(sentence.sentiment)
    print('--------------')

for sentence in blob_naive.sentences:
    print(sentence)
    print(sentence.sentiment)
    print('--------------')
