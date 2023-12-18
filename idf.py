from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def idf(chapters):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Compute TF-IDF vectors for chapters
    tfidf_matrix = vectorizer.fit_transform(chapters)

    # Compute similarity scores between chapters using cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Create a DataFrame to visualize similarity scores
    chapter_names = [f"Chapter {i+1}" for i in range(len(chapters))]
    similarity_df = pd.DataFrame(similarity_matrix, columns=chapter_names, index=chapter_names)

 

    plt.figure(figsize=(24, 18))
    sns.heatmap(similarity_df, annot=True, cmap="viridis")
    plt.title("Chapter Similarity based on TF-IDF Vectors")
    plt.xlabel("Chapters")
    plt.ylabel("Chapters")
    plt.show()