# http://stackoverflow.com/questions/34190860/sentiment-analysis-for-sentences-positive-negative-and-neutral
# http://textblob.readthedocs.io/en/dev/advanced_usage.html
import csv
import os
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from timeit import default_timer as timer

files_to_analyze = os.listdir("../test_data")
print(files_to_analyze)
# Open text file
with open('../test_data/moon.txt', 'r') as infile:
    EXAMPLE_TEXT = infile.read()

speaker_name = "Dr. Test"

### Using textblob
# blob_default = TextBlob(EXAMPLE_TEXT) # Using default PatternAnalyzer()
# blob_naive = TextBlob(EXAMPLE_TEXT, analyzer=NaiveBayesAnalyzer())

# for sentence in blob_default.sentences:
#     print(sentence)
#     print(sentence.sentiment)
#     print('--------------')

# for sentence in blob_naive.sentences:
#     print(sentence)
#     print(sentence.sentiment)
#     print('--------------')
    
### Using vader
# http://stackoverflow.com/questions/40325980/how-is-the-vader-compound-polarity-score-calculated-in-python-nltk
from nltk import tokenize
from nltk.tokenize import sent_tokenize

def demo_vader_instance(text):
    """
    Output polarity scores for a text using Vader approach.

    :param text: a text whose polarity has to be evaluated.
    """
    from nltk.sentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_scores = vader_analyzer.polarity_scores(text)
    print(vader_analyzer.polarity_scores(text))
    return vader_scores


# tokenize sentences
sentence_list = (sent_tokenize(EXAMPLE_TEXT))

with open("../outputs/test_2.csv", "wb") as csvfile:
  writer = csv.writer(csvfile, dialect="excel")
  writer.writerow(["sentence_number", "negative_score", "neutral_score", 
                    "positive_score", "compound_score", "processing_time", "text", "speaker_name"])
  sentence_number = 1
  
  for sentence in sentence_list:
    start = timer()
    vader_output = demo_vader_instance(sentence)
    end = timer()
    proc_time = end - start
    writer.writerow([sentence_number, vader_output['neg'], vader_output['neu'], 
                    vader_output['pos'], vader_output['compound'], proc_time, str(sentence), speaker_name])
    sentence_number += 1



    
