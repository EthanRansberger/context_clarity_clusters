import json
import os
import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from src.vectorizer import vectorize_text_count, vectorize_text_tfidf, vectorize_text_lda
from src.cluster_words import cluster_words, generate_file_structure, refine_clusters, visualize_clusters

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Function to load JSON data
def load_json(file_path):
    """
    Load JSON data from a file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The JSON data.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to save JSON data
def save_json(data, file_path):
    """
    Save JSON data to a file.

    Args:
        data (dict): The data to save.
        file_path (str): The path to the JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Function to preprocess text
def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase, removing punctuation,
    and tokenizing the text.

    Args:
        text (str): The text to preprocess.

    Returns:
        list: A list of cleaned and tokenized words.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stop words
    return tokens

# Extract user-authored content from JSON
def extract_user_authored_content(data):
    """
    Extract user-authored content from JSON data.

    Args:
        data (list): The JSON data.

    Returns:
        list: A list of user-authored content strings.
    """
    user_authored_content = []
    if not isinstance(data, list):
        return user_authored_content

    for record in data:
        mapping = record.get('mapping', {})
        if not isinstance(mapping, dict):
            continue

        for key, value in mapping.items():
            if not isinstance(value, dict):
                continue
            message = value.get('message')
            if not isinstance(message, dict):
                continue
            author = message.get('author')
            if not isinstance(author, dict) or author.get('role') != 'user':
                continue
            content = message.get('content')
            if not isinstance(content, dict):
                continue
            parts = content.get('parts')
            if isinstance(parts, list) and parts and isinstance(parts[0], str):
                user_authored_content.append(parts[0])
    return user_authored_content

def create_folder_tree(base_path, folder_count):
    """
    Create a nested folder tree based on the folder count.

    Args:
        base_path (str): The base directory where the folder tree will be created.
        folder_count (int): The number of folders to create.

    Returns:
        str: The path to the deepest folder created.
    """
    depth = math.ceil(math.log(folder_count, 10))
    path = base_path
    for d in range(depth):
        folder_index = (folder_count // (10 ** d)) % 10 + 1
        path = os.path.join(path, f'subfolder_{folder_index}')
        os.makedirs(path, exist_ok=True)
    return path

# Split content into a folder tree with a specified maximum number of chunks per folder
def split_content_into_folder_tree(content, output_dir, max_chunks_per_folder=10):
    """
    Split the content into a folder tree with a specified maximum number of chunks per folder.

    Args:
        content (list): The content to split.
        output_dir (str): The base directory where the folder tree will be created.
        max_chunks_per_folder (int): The maximum number of chunks per folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_content = len(content)
    total_chunks = math.ceil(total_content / max_chunks_per_folder)
    
    for i in range(total_chunks):
        chunk_start = i * max_chunks_per_folder
        chunk_end = min(chunk_start + max_chunks_per_folder, total_content)
        chunk = content[chunk_start:chunk_end]
        
        folder_path = create_folder_tree(output_dir, i)
        title = f'user_content_chunk_{i + 1}'
        chunk_file_path = os.path.join(folder_path, f'{title}.json')
        save_json(chunk, chunk_file_path)
        print(f'Saved {len(chunk)} strings to {chunk_file_path}')

# Generate a file structure based on clusters

# Main function to run the workflow
def main():
    input_file_path = 'data/conversations.json'  # Replace with your file path
    data = load_json(input_file_path)
    user_content = extract_user_authored_content(data)
    all_text = ' '.join(user_content)
    tokens = preprocess_text(all_text)
    tfidf_matrix, vectorizer = vectorize_text_tfidf(tokens)
    refine_clusters(tfidf_matrix, vectorizer)
    clusters = cluster_words(tfidf_matrix, vectorizer)
    visualize_clusters(clusters)

if __name__ == "__main__":
    main()
