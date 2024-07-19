from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

def vectorize_text_tfidf(documents):
    """
    Convert the preprocessed text tokens into numerical vectors using TF-IDF.

    Args:
        documents (list): A list of preprocessed text documents.

    Returns:
        tuple: A tuple containing the TF-IDF matrix and the vectorizer object.
    """
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        return tfidf_matrix, vectorizer
    except Exception as e:
        log_error(f"Error in TF-IDF vectorization: {str(e)}")
        return None, None

def vectorize_text_count(documents):
    """
    Convert the preprocessed text tokens into numerical vectors using Count Vectorizer.

    Args:
        documents (list): A list of preprocessed text documents.

    Returns:
        tuple: A tuple containing the Count Vectorizer matrix and the vectorizer object.
    """
    try:
        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(documents)
        return count_matrix, vectorizer
    except Exception as e:
        log_error(f"Error in Count Vectorizer vectorization: {str(e)}")
        return None, None

def vectorize_text_lda(documents, num_topics=10):
    """
    Convert the preprocessed text tokens into topic distributions using LDA.

    Args:
        documents (list): A list of preprocessed text documents.
        num_topics (int): The number of topics to find.

    Returns:
        tuple: A tuple containing the LDA matrix and the LDA model.
    """
    try:
        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(documents)
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
        lda_matrix = lda.fit_transform(count_matrix)
        return lda_matrix, lda
    except Exception as e:
        log_error(f"Error in LDA vectorization: {str(e)}")
        return None, None

def log_error(message):
    """Log an error message to a file."""
    with open("error_log.txt", "a") as log_file:
        log_file.write(message + "\n")
