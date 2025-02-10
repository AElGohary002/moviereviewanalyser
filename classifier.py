import pandas as pd 
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import word_tokenize
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("omw-1.4")

class NaiveBayes():
    

    # Create new Naive Bayes object storing number of classes and
    # selected features
    def __init__(self, number_classes, features) -> None:
        self.number_classes = number_classes
        self.features = features

    # Function that performs stemming on pandas dataframe
    def stemming(self,df):
        #Creating PorterStemmer object
        stemmer = SnowballStemmer('english')

        result = []

        for phrase in df:
            result.append(stemmer.stem(phrase))
        
        return result

    def lemmatization(self,text):

        result = []
        lemmatizer = WordNetLemmatizer()
    
        for token, tag in pos_tag(text):
            pos=tag[0]
            
            if pos not in ['a', 'r', 'n', 'v']:
                pos='n'
                
            result.append(lemmatizer.lemmatize(token,pos))
        return result
    
    # Function performing preprocessing techniques such as
    # lowercasing, tokenisation, and extracting POS tags
    def preprocess(self, df):

        #removing stopwords
        #stop_words = stopwords.words('english')

        #lowercasing
        df['Phrase'] = df['Phrase'].str.lower()

        #tokenisation
        
        def split(text):
            return text.split(' ')

        df['Phrase'] = df['Phrase'].apply(split)
        
        #df['Phrase']=df['Phrase'].apply(lambda X: word_tokenize(X))

        

        if self.features == "features" :          
            df['Phrase'] = df['Phrase'].apply(self.lemmatization)
            #POS Tagging
            df['Tagged_Phrase'] = df['Phrase'].apply(nltk.pos_tag)
        else:
            df['Phrase'] = df['Phrase'].apply(self.stemming)
       
        return df
    
    # Function that combines labels in the case that the user requests
    # a 3 class classifier
    def combine_classes(self, df):
        df['Sentiment'] = df['Sentiment'].replace([1], 0)
        df['Sentiment'] = df['Sentiment'].replace([2], 1)
        df['Sentiment'] = df['Sentiment'].replace([3], 2)
        df['Sentiment'] = df['Sentiment'].replace([4], 2)
        
    # Function that takes a dataset and label and returns its
    # prior probability
    def priors(self,train, sentiment):

        prior = self.sentiment_size(train, sentiment)/train.size

        return prior
    
    # Function that calculates the occurences of each word in the train set and returns
    # a nested dictionary of words and and their occurences in each label
    def occurences(self,train):
        occ = {}
        
        if self.features == 'all_words':
            for phrase, sentiment in zip(train.Phrase, train.Sentiment):
                for word in phrase:
                
                    if word not in occ:
                        # Adding word to occ if it is not already present
                        occ[word] ={}
                    else:
                        # Adding 1 to the value of the label that the word occurs in
                        occ[word][sentiment] = occ[word].get(sentiment, 0) +1
        else:
            # Iterating through a tuple of tagged phrases and sentiments as elements are tuples in that column
            for phrase, sentiment in zip(train.Tagged_Phrase, train.Sentiment):
                for word, tag in phrase:
                    # Feature extraction
                    if word not in occ and (tag =='JJ' or tag =='NN' or tag=='PDT' or tag == 'DT' or tag =='VB' or tag =='RB' or tag =='CC' or tag=='MD'):
                        occ[word]={}
                    elif (tag=='JJ' or tag =='NN' or tag == 'PDT' or tag == 'DT' or tag == 'VB' or tag == 'RB' or tag == 'CC' or tag =='MD'):
                        occ[word][sentiment] = occ[word].get(sentiment, 0) + 1
                    
        return occ
    
    # Function that calculates the frequency of a label in the training size
    def sentiment_size(self,train, sentiment):
        sentiments = []
        for label in train.Sentiment:
            if label == sentiment:
                sentiments.append(label)
        return len(sentiments)
    

    # Function that calculates the likelihood of each word given a label and returns a nested
    # dictionary similar to occurences but with the likelihood instead of occurence
    def likelihoods(self,occurences, train):
        likelihood = {}
        class_sizes={}

        # Computing different class sizes for different class configurations
        for i in range(self.number_classes):
            class_size =  self.sentiment_size(train, i)
            class_sizes[i] = class_size


        for word in occurences:
            likelihood[word]={}
            for sentiment in occurences[word]:
                    # Chaning the value of the label to likelihood instead of occurence
                    likelihood[word][sentiment] = (occurences[word][sentiment])/((class_sizes[sentiment]))
        
                        
        return likelihood

    # Function that takes the training and test sets, likelihoods, and number of classes
    # as input and returns a list of predicted_labels            
    def classify(self,train, dev, likelihoods, classes):
        predicted_labels = []
        
        for phrase in dev['Phrase']:
            posteriors = []
            
            for label in range(classes):
                # Calculating the prior probability for each class
                prior = self.priors(train, label)
                likelihood_array = []
               
                for word in phrase:
                    
                    try:
                        # Likelihood is added if current word is present both in dev and train
                        likelihood_array.append((likelihoods[word][label]))

                    # Exception for the case that word from dev is not present in train
                    except KeyError:
                        try:
                            if self.features == "features":
                                likelihood_array.append(min(likelihood_array)/10)
                            else:
                                likelihood_array.append(min(likelihood_array)/6)
                        # Exception for the case that a KeyError is caught with an emtpy likelihood array
                        except ValueError:
                            continue
                products = np.prod(likelihood_array)
                posterior =((products * prior))
                
                #Pairing posterior with label for each phrase
                posteriors.append((posterior, label))

            #Taking the maximum from the first element (posterior) and getting the corresponding label
            predicted_label = max(posteriors, key=lambda x:x[0])
            predicted_labels.append(predicted_label[1])
        return predicted_labels

    # Function that takes predicted labels and true labels and returns
    # three dictionaries  that contain false positives, false negatives, 
    # and true positives, respectively
    def evaluation(self,predicted_labels, actual_labels):
        fp = {}
        fn = {}
        tp = {}
       
        for pred, true in zip(predicted_labels, actual_labels):
            if pred != true:
                fp[pred] = fp.get(pred, 0)+1
                fn[true] = fn.get(true, 0)+1 
            else:
                tp[pred] = tp.get(pred, 0)+1 
        return fp, fn, tp
    
    # Function that calculates the macro-F1 score using the dictionaries obtained
    # from the evaluation function and returns the score
    def f1(self,predicted_labels, actual_labels):
        fps, fns, tps = self.evaluation(predicted_labels, actual_labels)
        
        f1_scores = []
        
        #Iterating number_classes times to find macro-F1 for each class
        for i in range(self.number_classes):
            sentiment = 2*tps[i]/(2*tps[i] + fps[i] + fns[i])
            f1_scores.append(sentiment)
        macro_f1 = (sum(f1_scores)/self.number_classes)

        return macro_f1
    
    # Function that constructs a confusion matrix and visualizes it using
    # matplotlib
    def cf_matrix(self,predicted_labels, actual_labels):
        
        # Converting dataframe column into a list
        actual_labels_list = list(actual_labels)
   
        data = {'y_actual':actual_labels_list,'y_pred': predicted_labels}
        df = pd.DataFrame(data)
        
        # Computing cross tabulation between dataframe columns
        confusion_matrix = pd.crosstab(df['y_actual'], df['y_pred'], rownames=['Actual'], colnames=['Predicted'])
        sns.set(color_codes=True)

        # Plotting confusion matrix
        sns.heatmap(confusion_matrix, annot=True, fmt='g')
        
        plt.show()