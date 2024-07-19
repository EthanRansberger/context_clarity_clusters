import os
import json
from src.text_preprocessor import preprocess_text
from src.vectorizer import vectorize_text_tfidf, vectorize_text_count, vectorize_text_lda
from src.utils.json_utils import load_json, save_json, extract_conversations_by_title, extract_user_authored_content
from src.utils.debug_utils import log_error, print_matrix_info
from src.cluster_words import cluster_words, generate_file_structure, visualize_clusters

def main():
    try:
        input_dir = os.path.join(os.path.dirname(__file__), 'data')
        output_dir = os.path.join(os.path.dirname(__file__), 'conversations')
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Extract and save all conversations by title from conversations.json
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(input_dir, filename)
                data = load_json(file_path)
                
                # Check if data is loaded
                if not data:
                    print(f"Failed to load data from JSON file: {filename}")
                    continue
                
                # Extract conversations by title and save them as separate chunks
                extract_conversations_by_title(data, output_dir=output_dir)

        # Step 2: Process each extracted conversation file for clustering
        for filename in os.listdir(output_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(output_dir, filename)
                data = load_json(file_path)
                
                # Check if data is loaded
                if not data:
                    print(f"Failed to load data from JSON file: {filename}")
                    continue
                
                user_content = extract_user_authored_content(data)
                
                # Check if user content is extracted
                if not user_content:
                    print(f"Extracted User Content is empty for {filename}.")
                    continue

                if not all(isinstance(item, str) for item in user_content):
                    log_error(f"All extracted user content must be strings in {filename}.")
                    continue
                
                preprocessed_texts = [preprocess_text(content) for content in user_content]
                
                # Check preprocessed texts
                if not preprocessed_texts:
                    print(f"Preprocessed texts are empty for {filename}.")
                    continue

                # Filter out documents with no tokens
                preprocessed_texts = [tokens for tokens in preprocessed_texts if len(tokens) > 0]

                if len(preprocessed_texts) < 2:
                    log_error(f"Not enough documents to perform clustering in {filename}. Ensure the preprocessing step is correct and there are enough user-authored content.")
                    continue

                documents = [' '.join(tokens) for tokens in preprocessed_texts]
                
                # Example usage with TF-IDF Vectorizer
                tfidf_matrix, tfidf_vectorizer = vectorize_text_tfidf(documents)
                if tfidf_matrix is None:
                    log_error(f"TF-IDF vectorization failed for {filename}.")
                    continue
                print_matrix_info(tfidf_matrix, "TF-IDF Matrix")

                # Example usage with Count Vectorizer
                count_matrix, count_vectorizer = vectorize_text_count(documents)
                if count_matrix is None:
                    log_error(f"Count Vectorizer vectorization failed for {filename}.")
                    continue
                print_matrix_info(count_matrix, "Count Matrix")

                # Example usage with LDA Vectorizer
                lda_matrix, lda_vectorizer = vectorize_text_lda(documents, num_topics=10)
                if lda_matrix is None:
                    log_error(f"LDA vectorization failed for {filename}.")
                    continue
                print_matrix_info(lda_matrix, "LDA Matrix")

                # Perform clustering and generate organized context documents
                num_clusters = max(2, min(5, tfidf_matrix.shape[0]))  # Ensure at least 2 clusters and at most the number of documents
                clusters = cluster_words(tfidf_matrix, tfidf_vectorizer, num_clusters)
                generate_file_structure(clusters, base_dir=os.path.join(os.path.dirname(__file__), 'output'))

                # Visualize clusters
                visualize_clusters(clusters)
        
    except Exception as e:
        log_error(f"Error in main workflow: {str(e)}")

if __name__ == "__main__":
    main()
