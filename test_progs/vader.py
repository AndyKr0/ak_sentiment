# http://www.nltk.org/_modules/nltk/sentiment/util.html
import sys
print(sys.version)

def demo_vader_instance(text):
    """
    Output polarity scores for a text using Vader approach.

    :param text: a text whose polarity has to be evaluated.
    """
    from nltk.sentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    print(vader_analyzer.polarity_scores(text))


def demo_vader_tweets(n_instances=None, output=None):
    """
    Classify 10000 positive and negative tweets using Vader approach.

    :param n_instances: the number of total tweets that have to be classified.
    :param output: the output file where results have to be reported.
    """
    from collections import defaultdict
    from nltk.corpus import twitter_samples
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.metrics import (accuracy as eval_accuracy, precision as eval_precision,
        recall as eval_recall, f_measure as eval_f_measure)

    if n_instances is not None:
        n_instances = int(n_instances/2)

    fields = ['id', 'text']
    positive_json = twitter_samples.abspath("positive_tweets.json")
    positive_csv = 'positive_tweets.csv'
    json2csv_preprocess(positive_json, positive_csv, fields, strip_off_emoticons=False,
                        limit=n_instances)

    negative_json = twitter_samples.abspath("negative_tweets.json")
    negative_csv = 'negative_tweets.csv'
    json2csv_preprocess(negative_json, negative_csv, fields, strip_off_emoticons=False,
                        limit=n_instances)

    pos_docs = parse_tweets_set(positive_csv, label='pos')
    neg_docs = parse_tweets_set(negative_csv, label='neg')

    # We separately split subjective and objective instances to keep a balanced
    # uniform class distribution in both train and test sets.
    train_pos_docs, test_pos_docs = split_train_test(pos_docs)
    train_neg_docs, test_neg_docs = split_train_test(neg_docs)

    training_tweets = train_pos_docs+train_neg_docs
    testing_tweets = test_pos_docs+test_neg_docs

    vader_analyzer = SentimentIntensityAnalyzer()

    gold_results = defaultdict(set)
    test_results = defaultdict(set)
    acc_gold_results = []
    acc_test_results = []
    labels = set()
    num = 0
    for i, (text, label) in enumerate(testing_tweets):
        labels.add(label)
        gold_results[label].add(i)
        acc_gold_results.append(label)
        score = vader_analyzer.polarity_scores(text)['compound']
        if score > 0:
            observed = 'pos'
        else:
            observed = 'neg'
        num += 1
        acc_test_results.append(observed)
        test_results[observed].add(i)
    metrics_results = {}
    for label in labels:
        accuracy_score = eval_accuracy(acc_gold_results,
            acc_test_results)
        metrics_results['Accuracy'] = accuracy_score
        precision_score = eval_precision(gold_results[label],
            test_results[label])
        metrics_results['Precision [{0}]'.format(label)] = precision_score
        recall_score = eval_recall(gold_results[label],
            test_results[label])
        metrics_results['Recall [{0}]'.format(label)] = recall_score
        f_measure_score = eval_f_measure(gold_results[label],
            test_results[label])
        metrics_results['F-measure [{0}]'.format(label)] = f_measure_score

    for result in sorted(metrics_results):
            print('{0}: {1}'.format(result, metrics_results[result]))

    if output:
        output_markdown(output, Approach='Vader', Dataset='labeled_tweets',
            Instances=n_instances, Results=metrics_results)




if __name__ == '__main__':
    from nltk.classify import NaiveBayesClassifier, MaxentClassifier
    from nltk.classify.scikitlearn import SklearnClassifier
    from nltk.tokenize import sent_tokenize
    from sklearn.svm import LinearSVC

    naive_bayes = NaiveBayesClassifier.train
    svm = SklearnClassifier(LinearSVC()).train
    maxent = MaxentClassifier.train

    #demo_tweets(naive_bayes)
    # demo_movie_reviews(svm)
    # demo_subjectivity(svm)
    # demo_sent_subjectivity("she's an artist , but hasn't picked up a brush in a year . ")
    # demo_liu_hu_lexicon("This movie was actually neither that funny, nor super witty.", plot=True)
    # demo_vader_instance("This movie was actually neither that funny, nor super witty.")
    # demo_vader_tweets()


    # My Code
    # tokenize text file into sentences
    with open('./test_data/my_sentences.txt', 'r') as infile:
        EXAMPLE_TEXT = infile.read()

    sentence_list = (sent_tokenize(EXAMPLE_TEXT))
    for sentence in sentence_list:
        print(sentence)
        print(demo_vader_instance(sentence))
        print('----------------------------')
