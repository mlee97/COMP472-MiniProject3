from gensim import corpora, models, similarities, downloader
from gensim.models import Word2Vec
import pandas as pd
import csv
import subprocess

# Runs Task1.py
subprocess.call("python Task1/Task1.py")

# Load same embedding size models.
wiki_model = downloader.load("glove-wiki-gigaword-300")         
ruscorpora_model = downloader.load("word2vec-ruscorpora-300")   

# Read the passed csv file.
test_set = pd.read_csv('synonyms.csv')

# Creates details file, which contains the question, answer, prediction, and label for each word in a test set.
#
# Parameters:
# - filename (String): The model name followed by "details.csv"
# - modelobj (Object): The variable name of the model object
def create_details_file(filename: str, modelobj):
    
    with open(filename, mode='w', newline='') as csv_file:    # Open a csv file.
        csv_writer = csv.writer(csv_file)
        correct_label_count = 0     # Stores the number of correct answers.
        guess_label_count = 0       # Stores the number of guesses.

        # Loop through each line in synonyms.csv.
        for i in range (0,len(test_set)):
            value = 0                   # Stores the greatest cosine similarity value.
            index_guess = '0'           # Stores the column title of the guess word.
            correct_guess = "wrong"     # Stores a boolean string value corresponding to whether the guess was correct or not.
    
            try:
                result = modelobj.similarity(test_set['question'][i], test_set['0'][i]) # Check the cosine similarity between the question word and the guess word in column '0'.
                if (result > value):    # If current cosine similarity ('result') is of greater value than 'value', assign 'value' to the bigger result number.
                    value = result  
                    index_guess = '0'
                    
                result = modelobj.similarity(test_set['question'][i], test_set['1'][i])
                if (result > value):
                    value = result
                    index_guess = '1'
                    
                result = modelobj.similarity(test_set['question'][i], test_set['2'][i])
                if (result > value):
                    value = result
                    index_guess = '2'
                    
                result = modelobj.similarity(test_set['question'][i], test_set['3'][i])
                if (result > value):
                    value = result
                    index_guess = '3'
                
                if(test_set['answer'][i] == test_set[index_guess][i]):  # Check if the guess is equal to the answer.
                    correct_guess = "correct"
                    correct_label_count = correct_label_count + 1

            except KeyError:    # Indicates that the question word was not in the wiki_model.
                correct_guess = "guess"     # Label as guess.
                guess_label_count = guess_label_count + 1

            csv_writer.writerow([test_set['question'][i], test_set['answer'][i], test_set[index_guess][i], correct_guess])  # Write a row to the csv file.
    
    return (correct_label_count, guess_label_count)


# Generate a CSV file comparing the results of using different models on the same test set.
#
# Parameters:
# - CE (String): Indicates the corpus and embedding identifier
# - modelname (String): The model name
# - modelobj (Object): The variable name of the model object
# - correct_label_count: The number of correct labels 
# - guess_label_count: The number of guess labels 
def create_analysis_file(CE: str, modelname: str, modelobj, correct_label_count, guess_label_count):
    
    # Output the parameters inside an analysis.csv file.
    with open('analysis.csv', mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        C = correct_label_count
        V = 80-guess_label_count
        
        if(V!=0): # Prevents division by 0.
            csv_writer.writerow([CE, modelname, len(modelobj), C, V, C/V])
        else:
            csv_writer.writerow([CE, modelname, len(modelobj), C, V, "Division by 0"])

# Generate details file for 2 new models from different corpora but same embedding size.
(wiki_correct, wiki_guess) = create_details_file('glove-wiki-gigaword-300-details.csv', wiki_model)
(ruscorpora_correct, ruscorpora_guess) = create_details_file('word2vec-ruscorpora-300-details.csv', ruscorpora_model)


# Append the 2 new models from different corpora but same embedding size to the analysis file.
create_analysis_file("C1-E1", "glove-wiki-gigaword-300", wiki_model, wiki_correct, wiki_guess)
create_analysis_file("C2-E2", "word2vec-ruscorpora-300", ruscorpora_model, ruscorpora_correct, ruscorpora_guess)


# Load 2 new models from the same corpus but different embedding sizes.
twitter_100_model = downloader.load("glove-twitter-100")
twitter_200_model = downloader.load("glove-twitter-200")


# Generate details file for 2 new models from the same corpus but different embedding sizes.
(twitter100_correct, twitter100_guess) = create_details_file('glove-twitter-100-details.csv', twitter_100_model)
(twitter200_correct, twitter200_guess) = create_details_file('glove-twitter-200-details.csv', twitter_200_model)


# Append the 2 new models from the same corpus but different embedding sizes to the analysis file.
create_analysis_file("C3-E3", "glove-twitter-100", twitter_100_model, twitter100_correct, twitter100_guess)
create_analysis_file("C4-E4", "glove-twitter-200", twitter_200_model, twitter200_correct, twitter200_guess)

