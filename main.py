# http://stackoverflow.com/questions/34190860/sentiment-analysis-for-sentences-positive-negative-and-neutral
# http://textblob.readthedocs.io/en/dev/advanced_usage.html
import csv
import os
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from timeit import default_timer as timer

source_dir = '/home/cabox/workspace/inagural'
files_to_analyze = os.listdir(source_dir)

for file in files_to_analyze:
  speaker_name = file.split('.')[0].split('-')[1]
  year = file.split('.')[0].split('-')[0]
  print("Analyzing " + file)
  # sentence number iterators i, j, and k
  i = j = k = 1

  # Open text file
  with open(source_dir + '/' + file, 'r') as infile:
      text_to_analyze = infile.read()

  ### Using textblob
  blob_default = TextBlob(text_to_analyze) # Using default PatternAnalyzer()
  print("Starting TextBlob default.")

  with open("./outputs/text_blob_default.csv", "ab") as csvfile:
    writer1 = csv.writer(csvfile, dialect="excel")
    
    # write headers to csv
    writer1.writerow(["sentence_number", "polarity", "subjectivity", 
                     "processing_time", "text", "speaker_name", "year"])

    for sentence in blob_default.sentences:
        sentence = sentence.replace("\n", "")
        start = timer()
        s = sentence.sentiment # (polarity, subjectivity)
        end = timer()
        proc_time = end - start
        writer1.writerow([i, s[0], s[1], proc_time, str(sentence), speaker_name, year])
        i += 1


  blob_naive = TextBlob(text_to_analyze, analyzer=NaiveBayesAnalyzer())
  print("Starting TextBlob Naive Bayes.")

  with open("./outputs/text_blob_naive.csv", "ab") as csvfile:
    writer2 = csv.writer(csvfile, dialect="excel")
    
    # write headers to csv
    writer2.writerow(["sentence_number", "polarity", "subjectivity", 
                     "processing_time", "text", "speaker_name", "year"])

    for sentence in blob_naive.sentences:

        sentence = sentence.replace("\n", "")
        start = timer()
        s = sentence.sentiment  # (polarity, subjectivity)
        end = timer()
        proc_time = end - start
        writer2.writerow([j, s[0], s[1], proc_time, str(sentence), speaker_name, year])
        j += 1


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
#       print(vader_analyzer.polarity_scores(text))
      return vader_scores


  # tokenize sentences
  sentence_list = (sent_tokenize(text_to_analyze))
  print("Starting Vader")
  with open("./outputs/vader.csv", "ab") as csvfile:
    writer3 = csv.writer(csvfile, dialect="excel")
    
    # write headers to csv
    writer3.writerow(["sentence_number", "negative_score", "neutral_score", 
                     "positive_score", "compound_score", "processing_time", 
                     "text", "speaker_name", "year"])
    

    for sentence in sentence_list:
      sentence = sentence.replace("\n", "")
      start = timer()
      vader_output = demo_vader_instance(sentence)
      end = timer()
      proc_time = end - start
      writer3.writerow([k, vader_output['neg'], vader_output['neu'], 
                      vader_output['pos'], vader_output['compound'], proc_time, str(sentence),
                      speaker_name, year])
      k += 1

    
