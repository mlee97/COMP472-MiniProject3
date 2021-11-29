from gensim import corpora, models, similarities, downloader
from gensim.models import Word2Vec
import pandas as pd
import csv

google_model = downloader.load("word2vec-google-news-300")  # Load the word2vec-google-news-300 pretrained embedding model. 
test_set = pd.read_csv('synonyms.csv')                      # Read the passed csv file.

with open('word2vec-google-news-300-details.csv', mode='w', newline='') as csv_file:    # Output in a csv file.
    csv_writer = csv.writer(csv_file)
    correct_label_count = 0 # Stores the numbner of correct answers.
    guess_label_count = 0   # Stores the number of guesses.

    # Loop through each line in synonyms.csv.
    for i in range (0,len(test_set)):
        value = 0                   # Stores the cosine similarity value.
        index_guess = '0'           # Stores the column title of the guess word.
        correct_guess = "wrong"     # Stores a boolean string value corresponding to whether the guess was correct or not.
 
        try:
            result = google_model.similarity(test_set['question'][i], test_set['0'][i]) # Check the cosine similarity between the question word and the guess word in column '0'.
            if (result > value):    # If current cosine similarity ('result') is of greater value than 'value', assign 'value' to the bigger result number.
                value = result  
                index_guess = '0'
                
            result = google_model.similarity(test_set['question'][i], test_set['1'][i])
            if (result > value):
                value = result
                index_guess = '1'
                
            result = google_model.similarity(test_set['question'][i], test_set['2'][i])
            if (result > value):
                value = result
                index_guess = '2'
                
            result = google_model.similarity(test_set['question'][i], test_set['3'][i])
            if (result > value):
                value = result
                index_guess = '3'
            
            if(test_set['answer'][i] == test_set[index_guess][i]):  # Check if the guess is equal to the answer.
                correct_guess = "correct"
                correct_label_count = correct_label_count + 1

        except KeyError:    # Indicates that the question word was not in the google_model.
            correct_guess = "guess"
            guess_label_count = guess_label_count + 1

        csv_writer.writerow([test_set['question'][i], test_set['answer'][i], test_set[index_guess][i], correct_guess])  # Write a row to the csv file.

# Output the parameters defined in Task 1 #2 inside an analysis.csv file.
with open('analysis.csv', mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    C = correct_label_count
    V = 80-guess_label_count
    csv_writer.writerow(["word2vec-google-news-300", len(google_model), C, V, C/V])