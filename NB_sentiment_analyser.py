# -*- coding: utf-8 -*-
import argparse
import csv
from classifier import NaiveBayes
import pandas as pd
"""
IMPORTANT, modify this part with your details
Name: Ali El Gohary
"""
USER_ID = "acb20ae" 

class Result_Store:
    def __init__(self, data):
        self.results = []
        self.data = data
    
    def store(self,predicted_labels):
        sentenceIdList = list(self.data["SentenceId"])
        for i in range(len(predicted_labels)):
            self.results.append("%s\t%s"%(sentenceIdList[i],predicted_labels[i]))

    def output(self, outfile):
        with open(outfile, 'w') as out:
            for predicted_labels in self.results:
                print(predicted_labels, file=out)

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args
    
    with open("file.tsv") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
    
    print(rd)


def main():
    
    inputs=parse_args()
    
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test

    train=pd.read_csv(training, sep='\t') 
    test = pd.read_csv(test, sep='\t')
    dev = pd.read_csv(dev, sep='\t')
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix

    
    classifier = NaiveBayes(number_classes, features)

    # Preprocessing
    for df in [train, dev, test]:
        classifier.preprocess(df)
        

    if number_classes == 3:
        classifier.combine_classes(train)
        classifier.combine_classes(dev)

    #occurences
    occs = classifier.occurences(train)

    #likelihoods
    likelihoods = classifier.likelihoods(occs,train)

    #classification
    predicted_labels = classifier.classify(train, dev, likelihoods, number_classes)
    test_pred = classifier.classify(train, test, likelihoods, number_classes)

    #evaluation
    true_labels = dev['Sentiment']
    f1_score = classifier.f1(predicted_labels, true_labels)

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #Storing and outputting predicted lables as tsv file
    if output_files == True:
        results = Result_Store(dev)
        results.store(predicted_labels)
        results.output(f"dev_predictions_{number_classes}classes_{USER_ID}.tsv")
        resultsTest = Result_Store(test)
        resultsTest.store(test_pred)
        resultsTest.output(f"test_predictions_{number_classes}classes_{USER_ID}.tsv")


    print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)")
    print("{:15}\t{:<10}\t{:<10}\t{:<10}".format(USER_ID, number_classes, features, f1_score))

    if confusion_matrix == True:
        classifier.cf_matrix(predicted_labels, true_labels)
    

if __name__ == "__main__":
    main()