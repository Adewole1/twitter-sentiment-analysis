import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import pandas as pd
import random

from statistics import mode

from sklearn.naive_bayes import BernoulliNB as BNB, MultinomialNB as MNB
from sklearn.linear_model import LogisticRegression as LR, SGDClassifier as SGDC
from sklearn.svm import SVC, LinearSVC

depressed = open("../Tagged/Tags/depressed.txt", "r", encoding='unicode_escape').read()
neutral = open("../Tagged/Tags/neutral.txt", "r", encoding='unicode_escape').read()
not_depressed = open("../Tagged/Tags/ndepressed.txt", "r", encoding='unicode_escape').read()

# print(all_data[0])