import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'WORD_EMBEDDINGS': 'starspace_embeddings.tsv',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    embeddings = {}
    for line in open(embeddings_path, encoding="utf-8"):
        line = line.strip().split('\t')
        embeddings[line[0]] = np.array(line[1:]).astype(np.float32)
    return embeddings, len(line) - 1


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    result = [embeddings[word] for word in question.split() if word in embeddings]
    return np.array(result).mean(axis=0) if result else np.zeros(dim)


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
