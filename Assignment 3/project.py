# Imports
import os
import re
import time
import string
import numpy as np
import pandas as pd

from scipy import sparse
from nltk.corpus import stopwords
from ordered_set import OrderedSet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Utility Functions
def clean_text(data):
    """ Cleans text by removing punctuation, normalizing whitespace, and converting to lowercase. """
    print("- Cleaning text data...")
    data = data.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    data = data.apply(lambda s: s.lower())
    data = data.apply(lambda s: re.sub(r'\s+', ' ', s))
    return data.drop_duplicates()

def tokenize_text(data):
    """ Tokenizes text into words. """
    print("- Tokenizing text data...")
    return data.apply(word_tokenize)

def remove_stopwords_and_short_words(tokens):
    """ Removes stopwords and short words from tokenized text. """
    print("- Removing stopwords and short words...")
    stop_words = set(stopwords.words('english'))
    return tokens.apply(lambda s: [word for word in OrderedSet(s) if word not in stop_words and len(word) > 2])

def stem_tokens(tokens):
    """ Stems tokens using a cache for efficiency. """
    print("- Stemming tokens...")
    ps = PorterStemmer()
    unique_tokens = np.unique(np.concatenate(tokens.values))
    stem_cache = {word: ps.stem(word) for word in unique_tokens}
    return tokens.apply(lambda s: [stem_cache[word] for word in s])

def generate_ngrams(tokens, n):
    """ Generates n-grams from tokens. """
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

# N-Gram Indexing
def create_ngram_indices(tokenized_data):
    """ Creates unigram and bigram indices from tokenized data. """
    print("- Creating unigram and bigram indices...")
    unigram_index, bigram_index = {}, {}

    for idx, tokens in enumerate(tokenized_data):
        # Generate and index unigrams
        for unigram in generate_ngrams(tokens, 1):
            unigram_index.setdefault(unigram, []).append(idx)

        # Generate and index bigrams
        for bigram in generate_ngrams(tokens, 2):
            bigram_index.setdefault(bigram, []).append(idx)

    return unigram_index, bigram_index

# TF-IDF Scoring
def apply_tfidf(cleaned_text):
    """ Applies TF-IDF vectorization and returns the vectorizer and DataFrame. """
    print("- Applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(use_idf=True)
    transformed_data = vectorizer.fit_transform(cleaned_text)
    tfidf_df = pd.DataFrame(transformed_data.toarray(), columns=vectorizer.get_feature_names_out())
    return vectorizer, tfidf_df

# BM25 Implementation
class BM25:
    def __init__(self, vectorizer, b=0.75, k1=1.6):
        self.vectorizer = vectorizer
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fits the model with the given data. """
        print("- Fitting BM25 model...")
        self.vectorizer.fit(X)
        self.y = sparse.csr_matrix(self.vectorizer.transform(X))
        self.avdl = self.y.sum(1).mean()

    def transform(self, q):
        """ Transforms query into BM25 scores. """
        print("- Calculating BM25 scores for query...")
        len_y = self.y.sum(1).A1
        query_vector = sparse.csr_matrix(self.vectorizer.transform([q]))
        y = self.y.tocsc()[:, query_vector.indices]
        denom = y + (self.k1 * (1 - self.b + self.b * len_y / self.avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, query_vector.indices] - 1.
        numer = y.multiply(np.broadcast_to(idf, y.shape)) * (self.k1 + 1)
        return (numer / denom).sum(1).A1

# Main Execution
if __name__ == "__main__":
    # Timer start
    start_time = time.time()
    
    # Data file path
    file_path = "../Week 1/resource/software_developer_united_states_1971_20191023_1.csv"

    # Step 1: Data Cleaning and Tokenization
    print("Reading and processing data...")
    raw_data = pd.read_csv(file_path)['job_description']
    cleaned_data = clean_text(raw_data)
    tokenized_data = tokenize_text(cleaned_data)

    # Step 2: Stopword Removal and Stemming
    filtered_tokens = remove_stopwords_and_short_words(tokenized_data)
    stemmed_tokens = stem_tokens(filtered_tokens)

    # Step 3: N-Gram Indexing
    unigram_index, bigram_index = create_ngram_indices(stemmed_tokens)

    # Step 4: TF-IDF Application
    tfidf_vectorizer, tfidf_df = apply_tfidf(cleaned_data)

    # Step 5: BM25 Calculation
    bm25 = BM25(tfidf_vectorizer)
    bm25.fit(cleaned_data)
    
    # Example Query
    query = 'cloud computing'
    bm25_scores = bm25.transform(query)

    # Step 6: Combine BM25 and N-Gram Matches
    print("Combining BM25 scores with n-gram matches...")
    query_tokens = query.split()
    unigram_matches = set().union(*(unigram_index.get((token,), []) for token in query_tokens))
    bigram_matches = set().union(*(bigram_index.get(bigram, []) for bigram in generate_ngrams(query_tokens, 2)))

    results = pd.DataFrame({
        'BM25_Score': bm25_scores,
        'Unigram_Match': [int(idx in unigram_matches) for idx in range(len(cleaned_data))],
        'Bigram_Match': [int(idx in bigram_matches) for idx in range(len(cleaned_data))]
    })

    results['Combined_Score'] = results.sum(axis=1)
    results = results.sort_values(by='Combined_Score', ascending=False)

    # Output
    print('------ OUTPUT ------')
    print('Top Results:')
    print(results.head())
    
    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Runtime executed : {elapsed_time}')
