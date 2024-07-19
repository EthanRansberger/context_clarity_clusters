import os
from sklearn.cluster import KMeans
from src.utils.debug_utils import log_error, print_matrix_info

def cluster_words(matrix, vectorizer, num_clusters=5):
    try:
        km = KMeans(n_clusters=num_clusters)
        km.fit(matrix)
        clustering = km.labels_.tolist()
        terms = vectorizer.get_feature_names_out() if hasattr(vectorizer, 'get_feature_names_out') else []
        clustered_terms = {i: [] for i in range(num_clusters)}
        
        for idx, cluster in enumerate(clustering):
            term = terms[idx] if idx < len(terms) else f"Topic_{idx}"
            clustered_terms[cluster].append(term)
        
        return clustered_terms
    except Exception as e:
        log_error(f"Error in cluster_words: {str(e)}")
        return {}

def generate_file_structure(clusters, base_dir='output'):
    try:
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
    except Exception as e:
        log_error(f"Error in generate_file_structure: {str(e)}")

def refine_clusters(matrix, vectorizer, num_iterations=5, initial_clusters=5):
    try:
        for iteration in range(num_iterations):
            clusters = cluster_words(matrix, vectorizer, initial_clusters + iteration)
            generate_file_structure(clusters, base_dir=f'output_iteration_{iteration}')
            # Additional analysis and feedback integration can be added here
    except Exception as e:
        log_error(f"Error in refine_clusters: {str(e)}")

def visualize_clusters(clusters):
    try:
        for cluster_id, terms in clusters.items():
            print(f'Cluster {cluster_id}: {", ".join(terms)}')
    except Exception as e:
        log_error(f"Error in visualize_clusters: {str(e)}")
