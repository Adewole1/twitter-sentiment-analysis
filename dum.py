from classifier import VoteClassifier, voted_classifier, training_set, testing_set
import pickle

if __name__ == '__main__':
    vote = VoteClassifier()
    features1 = open("pickled2/vote.pickle", "wb")
    pickle.dump(vote, features1)
    features1.close()
    
    save_classifier = open("pickled2\\voted_classifier.pickle", "wb")
    pickle.dump(voted_classifier, save_classifier)
    save_classifier.close()
    
    save_featureset = open("pickled2\\training_featureset.pickle", "wb")
    pickle.dump(training_set, save_featureset)
    save_featureset.close()

    save_featureset = open("pickled2\\testing_featureset.pickle", "wb")
    pickle.dump(testing_set, save_featureset)
    save_featureset.close()
    
    print('dumped')