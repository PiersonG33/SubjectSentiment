import os
import numpy as np
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import gensim.downloader as api

def get_average_sentiment(text, model):
    words = text.split()
    #print(len(model.key_to_index)) 

    word_sentiments = []

    for word in words:
        #print(word)
        if word in model.key_to_index:
            print(word)
            # Calculate sentiment based on polarity using word embeddings
            #print(word)
            blob = TextBlob(word)  # This can be adjusted to use embeddings directly
            polarity = blob.sentiment.polarity
            if(polarity != 0):
                word_sentiments.append(polarity)
            print(polarity)

    if (len(word_sentiments) == 0):
        word_sentiments.append(0)

    print("average Sentiment: ")
    print(np.mean(word_sentiments))

    return np.mean(word_sentiments) 

def perform_sentiment_analysis_new(CHATGBTshortText, model):
    total_sentiment_score = 0
    counter = 0

    for sentence in CHATGBTshortText:
        if sentence.strip():  # Skip empty sentences
            sentiment_score = get_average_sentiment(sentence, model)
            total_sentiment_score += sentiment_score
            counter += 1
    
    print("total_sentiment_scores: ")
    print(total_sentiment_score)

    print("average_sentiment_score: ")
    print(total_sentiment_score/counter)
    
    return total_sentiment_score/counter

def plotting_new(similarity_df, articleNames):
    # Ensure similarity_df is a list of values and articleNames is a list of article names
    plt.figure(figsize=(10, 6))
    sns.barplot(x=articleNames, y=similarity_df, palette='coolwarm')
    
    # Customize plot
    plt.title('Sentiment Analysis')
    plt.xlabel('Subjects')
    plt.ylabel('Polarity')
    plt.xticks(rotation=45, ha='right')  # Rotate x-ticks for better readability
    
    # Save and show the plot
    plt.tight_layout()  # Adjust layout to avoid overlapping
    plt.ylim(-1, 1)
    plt.axhline(y = 0, color = "black", linestyle = '-') 
    
    plt.savefig('plot.png')
    print(similarity_df)
    plt.show()


def main():

    #print("test")

    # uhhhhh wrong format
    #model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    info = api.info()  # show info about available models/datasets
    model = api.load("glove-twitter-25")  # download the model and return as object ready for use

    file = open("output.txt", 'r')
    text = file.read()

    sentiment_scores = perform_sentiment_analysis(text, model)
    plotting(sentiment_scores)

if __name__ == '__main__':
    main()
