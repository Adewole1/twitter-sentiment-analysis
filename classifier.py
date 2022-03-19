import random

import nltk
import collections
from nltk.classify.scikitlearn import SklearnClassifier as SKC
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from nltk.stem import (PorterStemmer, 
                       WordNetLemmatizer)
from nltk.corpus import (wordnet, 
                         stopwords)
from nltk.metrics.scores import (precision,
                                 recall,
                                 accuracy)
from statistics import mode
from sklearn.naive_bayes import (BernoulliNB as BNB, 
                                 MultinomialNB as MNB)
from sklearn.linear_model import (LogisticRegression as LR, 
                                  SGDClassifier as SGDC)
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle


# Class to 
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, featureset):
        votes = []
        for c in self._classifiers:
            v = c.classify(featureset)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, featureset):
        votes = []
        for c in self._classifiers:
            v = c.classify(featureset)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = (choice_votes/len(votes))*100
        return conf


# stops = set(stopwords.words("english"))

# functions to append sentence to list
# take each sentence and check if they are in allowed word types


# all_words = []
# documents = []

# parts of speech
# j = adjective
# r = adverb
# v = verb
# allowed_word_types = ["J", "JJ", "NN", "RB", "VB", "R", "V"]

# lemmatizer = WordNetLemmatizer()
# ps = PorterStemmer()

# open tagged documents for training and testing algorithms
# depressed = open("../Data/Tagged/Tags/depressed.txt", "r", encoding='unicode_escape').read()
# neutral = open("../Data/Tagged/Tags/neutral.txt", "r", encoding='unicode_escape').read()
# not_depressed = open("../Data/Tagged/Tags/ndepressed.txt", "r", encoding='unicode_escape').read()

# print('Started 1')
# for p in depressed.split('\n'):
#     documents.append( (p, "dep") )
#     words = word_tokenize(p)
#     pos = nltk.pos_tag(words)
#     for w in pos:
#         if w[1][0] not in stops:
#             all_words.append(w[0].lower())

# print('Started 2')           
# for p in neutral.split('\n'):
#     documents.append( (p, "neu") )
#     words = word_tokenize(p)
#     pos = nltk.pos_tag(words)
#     for w in pos:
#         if w[1][0] not in stops:
#             all_words.append(w[0].lower())

# print('Started 3')            
# for p in not_depressed.split('\n'):
#     documents.append( (p, "not") )
#     words = word_tokenize(p)
#     pos = nltk.pos_tag(words)
#     for w in pos:
#         if w[1][0] not in stops:
#             all_words.append(w[0].lower())
 
# print(len(all_words))
# print(len(documents))


# save_documents = open("pickled2\\documents.pickle", "wb")
# pickle.dump(documents, save_documents)
# save_documents.close()

# count = 0
# for w in all_words[:]:
#     all_words.append(ps.stem(w).lower())
#     all_words.append(lemmatizer.lemmatize(w).lower())
#     count += 1
#     print(count)

# save_all_words = open("pickled2\\all_words.pickle", "wb")
# pickle.dump(all_words, save_all_words)
# save_all_words.close()

# print(len(all_words))

# docs_f = open("pickled2\\documents.pickle", "rb")
# documents = pickle.load(docs_f)
# docs_f.close

# docs_f = open("pickled2\\all_words.pickle", "rb")
# all_words = pickle.load(docs_f)
# docs_f.close

# print(len(documents))
# print(len(all_words))
# print(all_words[:2])

# count = 0
# for w in all_words[:]:
#     count += 1
#     for syn in wordnet.synsets(w):
#         for l in syn.lemmas():
#             all_words.append(l.name().lower())
#             if l.antonyms():
#                 all_words.append(l.antonyms()[0].name().lower())
#     print(count)

# print(len(all_words))

# save_all_words = open("pickled2\\all_words_new.pickle", "wb")
# pickle.dump(all_words, save_all_words)
# save_all_words.close()

# print(len(all_words))
# print(len(documents))

# random.shuffle(documents)

# all_words = nltk.FreqDist(all_words)

# word_features = list(all_words.keys())
# print(len(word_features))
# print(len(all_words))

# save_all_words = open("pickled2\\word_features.pickle", "wb")
# pickle.dump(word_features, save_all_words)
# save_all_words.close()

# save_all_words = open("pickled\\all_words_freq.pickle", "wb")
# pickle.dump(all_words, save_all_words)
# save_all_words.close()

# docs_f = open("pickled2\\documents.pickle", "rb")
# documents = pickle.load(docs_f)
# docs_f.close

docs_f = open("pickled2\\word_features.pickle", "rb")
word_features = pickle.load(docs_f)
docs_f.close

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features


# featuresets = [(find_features(rev), category) for (rev, category) in documents]
# save_featureset = open("pickled2\\featureset.pickle", "wb")
# pickle.dump(featuresets, save_featureset)
# save_featureset.close()

# print(len(word_features))
# print(len(documents))
# print(len(featuresets))


# print(len(featuresets))
# print(featuresets[:5])
# random.shuffle(featuresets)


# all_words = nltk.FreqDist(all_words)

# word_features = list(all_words.keys())[:10000]


features = open("pickled2\\featureset.pickle", "rb")
featuresets = pickle.load(features)
features.close()

random.shuffle(featuresets)

# divide the featureset into training and testing sets
testing_set = featuresets[:]
training_set = featuresets[400:1400]

# save_featureset = open("pickled2\\training_featureset.pickle", "wb")
# pickle.dump(training_set, save_featureset)
# save_featureset.close()

# save_featureset = open("pickled2\\testing_featureset.pickle", "wb")
# pickle.dump(testing_set, save_featureset)
# save_featureset.close()

# features1 = open("pickled\\training_featureset.pickle", "rb")
# training_set = pickle.load(features1)
# features1.close()

# features2 = open("pickled\\testing_featureset.pickle", "rb")
# testing_set = pickle.load(features2)
# features2.close()


# Train and test algorithms
# Along with their accuracy

# Naive Bayes classifier
# classifier_NB = nltk.NaiveBayesClassifier.train(training_set)
# print("Naive Bayes Original Algorithm accuracy: ", (nltk.classify.accuracy(classifier_NB, testing_set))*100)
# classifier_NB.show_most_informative_features(10)
# print("Naive Bayes Original Algorithm Precision: ", (precision(training_set, testing_set)))
# print("Naive Bayes Original Algorithm Recall: ", (recall(training_set, testing_set)))

# Multinomial NAive Bayes classifier
# MNB_classifier = SKC(MNB())
# MNB_classifier.train(training_set)
# print("Multinomial_NB classifier accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
# MNB_classifier.show_most_informative_features(10)
# print("Multinomial_NB Algorithm Precision: ", (precision(MNB_classifier, testing_set)))
# print("Multinomial_NB Algorithm Recall: ", (recall(MNB_classifier, testing_set)))


# Bernoulli Naive Bayes classifier
# BNB_classifier = SKC(BNB())
# BNB_classifier.train(training_set)
# print("Bernoulli_NB classifier accuracy: ", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)
# BNB_classifier.show_most_informative_features(10)
# print("Bernoulli_NB classifier Algorithm Precision: ", (precision(BNB_classifier, testing_set)))
# print("Bernoulli_NB classifier Algorithm Recall: ", (recall(BNB_classifier, testing_set)))


# Logistic regression classifier
# LR_classifier = SKC(LR())
# LR_classifier.train(training_set)
# print("LogisticRregression classifier accuracy: ", (nltk.classify.accuracy(LR_classifier, testing_set))*100)
# LR_classifier.show_most_informative_features(10)
# print("LogisticeRegression classifier Algorithm Precision: ", (precision(LR_classifier, testing_set)))
# print("LogisticeRegression classifier Algorithm Recall: ", (recall(LR_classifier, testing_set)))



# LinearSVC classifier
# LinearSVC_classifier = SKC(LinearSVC())
# LinearSVC_classifier.train(training_set)
# print("LinearSVC classifier accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
# LinearSVC_classifier.show_most_informative_features(10)
# print("LinearSVC classifier Algorithm Precision: ", (precision(LinearSVC_classifier, testing_set)))
# print("LinearSVC classifier Algorithm Recall: ", (recall(LinearSVC_classifier, testing_set)))


# SGDC classifier
# SGDC_classifier = SKC(SGDC())
# SGDC_classifier.train(training_set)
# print("SDGC classifier accuracy: ", (nltk.classify.accuracy(SGDC_classifier, testing_set))*100)
# SGDC_classifier.show_most_informative_features(10)
# print("SGDC classifier Algorithm Precision: ", (precision(SGDC_classifier, testing_set)))
# print("SGDC classifier Algorithm Recall: ", (recall(SGDC_classifier, testing_set)))


# SVC classifier
# SVC_classifier = SKC(SVC())
# SVC_classifier.train(training_set)
# print("SVC classifier accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
# SVC_classifier.show_most_informative_features(10)
# print("SVC classifier Algorithm Precision: ", (precision(SVC_classifier, testing_set)))
# print("SVC classifier Algorithm Recall: ", (recall(SVC_classifier, testing_set)))



# classifier = nltk.DecisionTreeClassifier.train(training_set)
# classifier.train(training_set)
# print("Decision tree classifier accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
# classifier.show_most_informative_features(10)
# print("Decision Tree Original Algorithm Precision: ", (precision(classifier, testing_set)))
# print("Decision Tree Original Algorithm Recall: ", (recall(classifier, testing_set)))


# run throough all classifiers and vote for one
# voted_classifier = VoteClassifier(
#     classifier_NB,
#     LinearSVC_classifier,
#     MNB_classifier,
#     BNB_classifier,
#     LR_classifier,
#     SGDC_classifier,
#     SVC_classifier,
#     # classifier,
# )

# save_classifier = open("pickled2\\voted_classifier.pickle", "wb")
# pickle.dump(voted_classifier, save_classifier)
# save_classifier.close()

# refsets = collections.defaultdict(set)
# testsets = collections.defaultdict(set)

# for i, (feats, label) in enumerate(testing_set):
#     refsets[label].add(i)
#     observed = voted_classifier.classify(feats)
#     testsets[observed].add(i)
    
# save_classifier = open("pickled2\\ref_set.pickle", "wb")
# pickle.dump(refsets, save_classifier)
# save_classifier.close()

# save_classifier = open("pickled2\\test_set.pickle", "wb")
# pickle.dump(testsets, save_classifier)
# save_classifier.close()


features1 = open("pickled2\\voted_classifier.pickle", "rb")
voted_classifier = pickle.load(features1)
features1.close()

features1 = open("pickled2\\ref_set.pickle", "rb")
refsets = pickle.load(features1)
features1.close()

features1 = open("pickled2\\test_set.pickle", "rb")
testsets = pickle.load(features1)
features1.close()


print(len(refsets), len(testsets))
# print("Voted classifier Accuracy: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
print('Voted classifier Precision:', precision(refsets['dep'], testsets['dep'])*100)
print('Voted classifier Recall:', recall(refsets['dep'], testsets['dep'])*100)
# print('Voted classifier Accuracy:', accuracy(refsets['dep'], testsets['dep'])*100)

# function to classify tweet, showing the confidence of the classification
def  sentiment(text):
    feats = find_features(text)
    
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

# '''

print(sentiment('I am tired of this life, very tired'))