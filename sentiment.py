import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt


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

def model_training_test(text, labels):
    df = pd.DataFrame({'review': text, 'label': labels})
    tokenized_reviews = [review.split() for review in df['review']]
    model = Word2Vec(sentences=tokenized_reviews, vector_size=300, window=5, min_count=1, workers=4)
    
    return model

def plotting(similarity_df):
# Visualize the similarities
    sns.barplot(x=similarity_df.index, y='correlation', data=similarity_df.reset_index())
    plt.title('Project of Projects')
    plt.xlabel('Articles')
    plt.ylabel('Potential Biases')
    plt.xticks(rotation=45)
    plt.show()