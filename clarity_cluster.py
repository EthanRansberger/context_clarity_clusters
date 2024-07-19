import json
from src.vectorizer import vectorize_text_tfidf, vectorize_text_count, vectorize_text_lda
from src.cluster_words import cluster_words, generate_file_structure, refine_clusters, visualize_clusters
from src.json_loader import load_json, extract_conversations_by_title, extract_user_authored_content
from src.text_preprocessor import preprocess_text

# Main function to run the workflow
def main():
    input_file_path = 'data/conversations.json'  # Replace with your file path
    data = load_json(input_file_path)
    
    # Extract conversations by title
    extract_conversations_by_title(data, output_dir='conversations')
    
    user_content = extract_user_authored_content(data)
    
    if not all(isinstance(item, str) for item in user_content):
        raise ValueError("All extracted user content must be strings.")
    
    all_text = ' '.join(user_content)  # Join list of strings into a single string
    tokens = preprocess_text(all_text)

    # Example usage with TF-IDF Vectorizer
    tfidf_matrix, tfidf_vectorizer = vectorize_text_tfidf(tokens)
    num_samples = tfidf_matrix.shape[0]
    num_clusters = max(2, min(5, num_samples))  # Ensure the number of clusters is >= 2 and <= number of samples
    refine_clusters(tfidf_matrix, tfidf_vectorizer, initial_clusters=num_clusters)
    clusters = cluster_words(tfidf_matrix, tfidf_vectorizer, num_clusters=num_clusters)
    visualize_clusters(clusters)

    # Example usage with Count Vectorizer
    count_matrix, count_vectorizer = vectorize_text_count(tokens)
    num_samples = count_matrix.shape[0]
    num_clusters = max(2, min(5, num_samples))  # Ensure the number of clusters is >= 2 and <= number of samples
    refine_clusters(count_matrix, count_vectorizer, initial_clusters=num_clusters)
    clusters = cluster_words(count_matrix, count_vectorizer, num_clusters=num_clusters)
    visualize_clusters(clusters)

    # Example usage with LDA Vectorizer
    lda_matrix, lda_vectorizer = vectorize_text_lda(tokens, num_topics=10)
    num_samples = lda_matrix.shape[0]
    num_clusters = max(2, min(5, num_samples))  # Ensure the number of clusters is >= 2 and <= number of samples
    refine_clusters(lda_matrix, lda_vectorizer, initial_clusters=num_clusters)
    clusters = cluster_words(lda_matrix, lda_vectorizer, num_clusters=num_clusters)
    visualize_clusters(clusters)

if __name__ == "__main__":
    main()
