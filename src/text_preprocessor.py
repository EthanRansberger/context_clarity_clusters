import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Ensure you have the necessary NLTK resources downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Initialize stop words and stemmer/lemmatizer
stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def normalize_text(text):
    """Normalize the text by converting to lowercase and removing special characters."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    return text

def tokenize_text(text):
    """Tokenize the text into words."""
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    """Remove stopwords from the list of tokens."""
    return [token for token in tokens if token not in stop_words]

def stem_tokens(tokens):
    """Stem the tokens."""
    return [stemmer.stem(token) for token in tokens]

def lemmatize_tokens(tokens):
    """Lemmatize the tokens."""
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text, use_stemming=False, use_lemmatization=True):
    """Preprocess the input text with options for stemming and lemmatization."""
    try:
        text = normalize_text(text)
        tokens = tokenize_text(text)
        tokens = remove_stopwords(tokens)
        if use_stemming:
            tokens = stem_tokens(tokens)
        if use_lemmatization:
            tokens = lemmatize_tokens(tokens)
        return tokens
    except Exception as e:
        log_error(f"Error in preprocessing text: {str(e)}")
        return []

def log_error(message):
    """Log an error message to a file."""
    with open("error_log.txt", "a") as log_file:
        log_file.write(message + "\n")
