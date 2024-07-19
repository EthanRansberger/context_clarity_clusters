import os
from sklearn.cluster import KMeans

def cluster_words(matrix, vectorizer, num_clusters=5):
    """
    Apply K-Means clustering to the matrix to group similar words or phrases.

    Args:
        matrix (scipy sparse matrix or numpy array): The matrix of text vectors.
        vectorizer (object): The vectorizer object used to create the matrix.
        num_clusters (int): The number of clusters to form.

    Returns:
        dict: A dictionary with cluster IDs as keys and lists of terms as values.
    """
    km = KMeans(n_clusters=num_clusters)
    km.fit(matrix)
    clustering = km.labels_.tolist()
    terms = vectorizer.get_feature_names_out() if hasattr(vectorizer, 'get_feature_names_out') else []
    clustered_terms = {i: [] for i in range(num_clusters)}
    
    for idx, cluster in enumerate(clustering):
        term = terms[idx] if idx < len(terms) else f"Topic_{idx}"
        clustered_terms[cluster].append(term)
    
    return clustered_terms

def generate_file_structure(clusters, base_dir='output'):
    """
    Generate a file structure based on clusters, creating directories and text files for each cluster and term.

    Args:
        clusters (dict): A dictionary with cluster IDs as keys and lists of terms as values.
        base_dir (str): The base directory where the file structure will be created.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for cluster_id, terms in clusters.items():
        cluster_dir = os.path.join(base_dir, f'cluster_{cluster_id}')
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        
        for term in terms:
            file_path = os.path.join(cluster_dir, f'{term}.txt')
            with open(file_path, 'w') as file:
                file.write(f'Term: {term}\nCluster ID: {cluster_id}\n')
    
    print(f'File structure generated at {base_dir}')

def refine_clusters(matrix, vectorizer, num_iterations=5, initial_clusters=5):
    """
    Run multiple iterations to refine clusters and generate file structures.

    Args:
        matrix (scipy sparse matrix): The matrix of text vectors.
        vectorizer (object): The vectorizer object used to create the matrix.
        num_iterations (int): The number of iterations to run.
        initial_clusters (int): The initial number of clusters.
    """
    for iteration in range(num_iterations):
        clusters = cluster_words(matrix, vectorizer, initial_clusters + iteration)
        generate_file_structure(clusters, base_dir=f'output_iteration_{iteration}')
        # Additional analysis and feedback integration can be added here

def visualize_clusters(clusters):
    """
    Print the clusters for visualization.

    Args:
        clusters (dict): A dictionary with cluster IDs as keys and lists of terms as values.
    """
    for cluster_id, terms in clusters.items():
        print(f'Cluster {cluster_id}: {", ".join(terms)}')
