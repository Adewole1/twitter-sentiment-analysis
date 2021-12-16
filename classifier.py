import random

import nltk
from nltk.classify.scikitlearn import SklearnClassifier as SKC
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from nltk.stem import (PorterStemmer, WordNetLemmatizer)
from nltk.corpus import wordnet
from nltk.metrics.scores import (precision, recall)

from statistics import mode

from sklearn.naive_bayes import BernoulliNB as BNB, MultinomialNB as MNB
from sklearn.linear_model import (LogisticRegression as LR, SGDClassifier as SGDC)
from sklearn.svm import SVC, LinearSVC
# from sklearn.metrics import precision_score, recall_score, f1_score


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





# functions to append sentence to list
# take each sentence and check if they are in allowed word types
'''

all_words = []
documents = []

# parts of speech
# j = adjective
# r = adverb
# v = verb
allowed_word_types = ["J", "JJ", "NN", "RB", "VB", "R", "V"]

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

# open tagged documents for training and testing algorithms
depressed = open("../Tagged/Tags/depressed.txt", "r", encoding='unicode_escape').read()
neutral = open("../Tagged/Tags/neutral.txt", "r", encoding='unicode_escape').read()
not_depressed = open("../Tagged/Tags/ndepressed.txt", "r", encoding='unicode_escape').read()


for p in depressed.split('\n'):
    documents.append( (p, "dep") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            
for p in neutral.split('\n'):
    documents.append( (p, "neu") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            
for p in not_depressed.split('\n'):
    documents.append( (p, "not") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
 
# print(len(all_words))

# save_all_words = open("all_words.pickle", "wb")
# pickle.dump(all_words, save_all_words)
# save_all_words.close()


# save_documents = open("documents.pickle", "wb")
# pickle.dump(documents, save_documents)
# save_documents.close()
'''


words_f = open("Pickled\\all_words_new.pickle", "rb")
all_word = pickle.load(words_f)
words_f.close

docs_f = open("Pickled\\documents.pickle", "rb")
documents = pickle.load(docs_f)
docs_f.close

'''

for w in all_words:
    all_words.append(ps.stem(w).lower())
    all_words.append(lemmatizer.lemmatize(w).lower())

save_all_words = open("all_words_new.pickle", "wb")
pickle.dump(all_words, save_all_words)
save_all_words.close()

print(len(all_words))


for w in all_words[8000:9094]:
    for syn in wordnet.synsets(w):
        for l in syn.lemmas():
            all_words.append(l.name().lower())
            if l.antonyms():
                all_words.append(l.antonyms()[0].name().lower())


# save_all_words = open("all_words_new.pickle", "wb")
# pickle.dump(all_words, save_all_words)
# save_all_words.close()

random.shuffle(documents)

all_words = nltk.FreqDist(all_word)

word_features = list(all_words.keys())[:10000]



featuresets = [(find_features(rev), category) for (rev, category) in documents]
save_featureset = open("featureset.pickle", "wb")
pickle.dump(featuresets, save_featureset)
save_featureset.close()


# print(len(documents))


# print(len(featuresets))
# print(featuresets[:5])
# random.shuffle(featuresets)
'''

all_words = nltk.FreqDist(all_word)

word_features = list(all_words.keys())[:10000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features

features = open("Pickled/featureset.pickle", "rb")
featuresets = pickle.load(features)
features.close()

# divide the featureset into training and testing sets
testing_set = featuresets[:400]
training_set = featuresets[400:]


# '''
# Train and test algorithms
# Along with their accuracy

# Naive Bayes classifier
classifier_NB = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Original Algorithm accuracy: ", (nltk.classify.accuracy(classifier_NB, testing_set))*100)
# classifier.show_most_informative_features(15)
print("Decision Tree Original Algorithm Precision: ", (precision(classifier_NB, testing_set)))
print("Decision Tree Original Algorithm Recall: ", (recall(classifier_NB, testing_set)))

# Multinomial NAive Bayes classifier
MNB_classifier = SKC(MNB())
MNB_classifier.train(training_set)
print("Multinomial_NB classifier accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)



# Bernoulli Naive Bayes classifier
BNB_classifier = SKC(BNB())
BNB_classifier.train(training_set)
print("Bernoulli_NB classifier accuracy: ", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)



# Logistic regression classifier
LR_classifier = SKC(LR())
LR_classifier.train(training_set)
print("LogisticRregression classifier accuracy: ", (nltk.classify.accuracy(LR_classifier, testing_set))*100)



# LinearSVC classifier
LinearSVC_classifier = SKC(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC classifier accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


# SGDC classifier
SGDC_classifier = SKC(SGDC())
SGDC_classifier.train(training_set)
print("SDGC classifier accuracy: ", (nltk.classify.accuracy(SGDC_classifier, testing_set))*100)


# SVC classifier
SVC_classifier = SKC(SVC())
SVC_classifier.train(training_set)
print("SVC classifier accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

# classifier = nltk.DecisionTreeClassifier.train(training_set)

# run throough all classifiers and vote for one
voted_classifer = VoteClassifier(
    classifier_NB,
    LinearSVC_classifier,
    MNB_classifier,
    BNB_classifier,
    LR_classifier,
    SGDC_classifier,
    SVC_classifier
)
print("Voted classifier accuracy: ", (nltk.classify.accuracy(voted_classifer, testing_set))*100)


# function to classify tweet, showoing the confidence of the classification
def  sentiment(text):
    feats = find_features(text)
    
    return voted_classifer.classify(feats), voted_classifer.confidence(feats)

# '''