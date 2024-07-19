from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def vectorize_text_tfidf(tokens):
    """
    Convert the preprocessed text tokens into numerical vectors using TF-IDF.

    Args:
        tokens (list): A list of preprocessed text tokens.

    Returns:
        tuple: A tuple containing the TF-IDF matrix and the vectorizer object.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(tokens)])
    return tfidf_matrix, vectorizer

def vectorize_text_count(tokens):
    """
    Convert the preprocessed text tokens into numerical vectors using Count Vectorizer.

    Args:
        tokens (list): A list of preprocessed text tokens.

    Returns:
        tuple: A tuple containing the Count Vectorizer matrix and the vectorizer object.
    """
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform([' '.join(tokens)])
    return count_matrix, vectorizer

def vectorize_text_lda(tokens, num_topics=10):
    """
    Convert the preprocessed text tokens into topic distributions using LDA.

    Args:
        tokens (list): A list of preprocessed text tokens.
        num_topics (int): The number of topics to find.

    Returns:
        tuple: A tuple containing the LDA matrix and the LDA model.
    """
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform([' '.join(tokens)])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda_matrix = lda.fit_transform(count_matrix)
    return lda_matrix, lda
