'''
    To run this file, ensure you have saved the pre_train_prior_likelihood.txt file in the same folder, which is created by
    CS6120_NLP_Assignment_1_Notebook.ipynb.
    Launch cmd and run 'python test_classifier.py'
'''

# Importing the Libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
import string
import os
warnings.filterwarnings("ignore")

def clean_review(review):
    '''
    Input:
        review: a string containing a review.
    Output:
        review_cleaned: a processed review.

    '''
    # convert to lowercase
    review = review.lower()

    # remove links i.e. <br> tags
    review = re.sub("<br />", " ", review)

    # remove punctuations
    review = review.translate(str.maketrans('', '', string.punctuation))

    # remove stopwords
    word_tokens = word_tokenize(review)
    stop_words = set(stopwords.words('english'))
    word_tokens_without_sw = [word for word in word_tokens if not word in stop_words]

    # concatenate word tokens without sw to form a sentence
    filtered_words = (" ").join(word_tokens_without_sw)

    # stemming operation
    word_tokens = word_tokenize(filtered_words)
    lemmatizer = WordNetLemmatizer()
    singles = [lemmatizer.lemmatize(word) for word in word_tokens]

    # concatenate word tokens back to form a sentence
    review_cleaned = (" ").join(singles)

    return review_cleaned

def naive_bayes_predict(review, logprior, loglikelihood):
    '''
    Params:
        review: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Return:
        total_prob: the sum of all the loglikelihoods of each word in the review (if found in the dictionary) + logprior (a number)

    '''
    # process the review to get a list of words
    word_l = word_tokenize(clean_review(review))

    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob += logprior

    for word in word_l:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            total_prob += loglikelihood.get(word)

    if total_prob > 0:
        total_prob = 1
    else:
        total_prob = 0

    return total_prob

def pre_train_model():
    '''
    loads pre-trained model

    Params: None
    Return:
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers

    '''
    logprior = 0
    loglikelihood = {}

    # opens file
    f = open("pre_train_prior_likelihood.txt", "r", encoding="utf8")

    # read in logprior value
    logprior = float(f.readline())

    # reads rest of the file for loglikelihood values
    for line in f.readlines():
        split = line.split(",")
        loglikelihood[split[0]] = float(split[1])

    return logprior, loglikelihood

def print_results(review, decision, loglikelihood):
    '''
    a util method to print classifier results for the given review

    Params: None
    Return:
        prints probabilities for each input token along with the final classification decision

    '''
    # tokenize the review
    tokens = word_tokenize(clean_review(review))

    # print probability of each token
    for token in tokens:
        print("Given token: ", token, ", Probability: ", loglikelihood.get(token))

    # print the sentiment classification
    print("\nClassification of the review: ", decision,  "\n")

if __name__ == "__main__":

    while True:
        # takes in user input
        review = input("Please provide a review to classify or enter 'X' to quit the classifier application: \n")

        # check if the given input is 'X'
        if review.upper() == 'X':
            break
        else:
            # fetch logprior and loglikelihood values
            logprior, loglikelihood = pre_train_model()

            # call predict function for the given user review
            decision = naive_bayes_predict(review, logprior, loglikelihood)

            # map it to correct sentiment class
            if decision == 0:
                decision = 'Positive'
            else:
                decision = 'Negative'

            # print results
            print_results(review, decision, loglikelihood)
