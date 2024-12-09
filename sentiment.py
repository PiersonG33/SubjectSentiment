import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import gensim.downloader as api

def get_average_embedding(words, model):
    embeddings = []
    for word in words:
        if word in model.wv.key_to_index:
            embeddings.append(model.wv[word])
    return np.mean(embeddings, axis=0) if embeddings else None


def simularity_calc(position_embeddings, positive_embedding, negative_embedding):
    similarities = {}
    for position, embedding in position_embeddings.items():
        if embedding is not None:
            pos_sim = cosine_similarity([embedding], [positive_embedding])[0][0]
            neg_sim = cosine_similarity([embedding], [negative_embedding])[0][0]
            similarities[position] = {'positive_similarity': pos_sim, 'negative_similarity': neg_sim}

    similarity_df = pd.DataFrame(similarities).T
    similarity_df['correlation'] = similarity_df['positive_similarity'] - similarity_df['negative_similarity']

    return similarity_df['correlation']

'''
def model_training_test(text, labels):
    df = pd.DataFrame({'review': text, 'label': labels})
    tokenized_reviews = [review.split() for review in df['review']]
    model = Word2Vec(sentences=tokenized_reviews, vector_size=300, window=5, min_count=1, workers=4)
    
    return model
'''

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

def perform_sentiment_analysis(text, model):
    sentences = text.split('\n')  # Split the text into sentences
    sentiment_scores = []

    for sentence in sentences:
        if sentence.strip():  # Skip empty sentences
            sentiment_score = get_average_sentiment(sentence, model)
            sentiment_scores.append(sentiment_score)
    
    return sentiment_scores

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

def plotting(similarity_df):
# Visualize the similarities
    plt.figure(figsize=(10, 6))
    sns.barplot(x=np.arange(len(similarity_df)), y=similarity_df, palette='coolwarm')
    #sns.barplot(x=similarity_df.index, y='correlation', data=similarity_df.reset_index())
    plt.title('Sentiment Analysis')
    plt.xlabel('Various Articles (needs to be labeled)')
    plt.ylabel('Polarity')
    plt.xticks(rotation=45)
    plt.savefig('plot.png')

    print(similarity_df)
    plt.show()

def plotting_new(similarity_df, articleNames):
    # Ensure similarity_df is a list of values and articleNames is a list of article names
    plt.figure(figsize=(10, 6))
    sns.barplot(x=articleNames, y=similarity_df, palette='coolwarm')
    
    # Customize plot
    plt.title('Sentiment Analysis')
    plt.xlabel('Various Articles')
    plt.ylabel('Polarity')
    plt.xticks(rotation=45, ha='right')  # Rotate x-ticks for better readability
    
    # Save and show the plot
    plt.tight_layout()  # Adjust layout to avoid overlapping
    plt.savefig('plot.png')
    print(similarity_df)
    plt.show()

# loads the model
def load_model(text_data=None, modelname=None):
    if modelname is not None:
        # Load the pre-trained model from the provided path
        model = Word2Vec.load(modelname)
        print("Loaded pre-trained Word2Vec model.")
    elif text_data is not None:
        # If no modelname is given, and text_data is provided, train a new model
        print("Training a new Word2Vec model from the provided text data...")
        df = pd.DataFrame({'review': text, 'label': labels})
        tokenized_reviews = [review.split() for review in df['review']]
        model = Word2Vec(sentences=tokenized_reviews, vector_size=300, window=5, min_count=1, workers=4)
        print("Training complete. New model created.")
    else:
        # If neither modelname nor text is provided, return a default empty Word2Vec model
        print("No model or text provided. Returning a default Word2Vec model.")
        model = Word2Vec(vector_size=300, window=5, min_count=1, workers=4)  # Default empty model
    
    return model

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
