# Imports
import os
import re
import string
import numpy as np
import pandas as pd

from scipy import sparse
from nltk.corpus import stopwords
from ordered_set import OrderedSet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Functions
def get_and_clean_data(file_path):
    print('Reading data file...')
    data = pd.read_csv(file_path)
    description = data['job_description']
    
    # Remove punctuation, non-breaking spaces, and make lowercase
    cleaned_description = description.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    cleaned_description = cleaned_description.apply(lambda s: s.lower())
    
    # Normalize whitespace
    cleaned_description = cleaned_description.apply(lambda s: s.translate(str.maketrans(string.whitespace, ' '*len(string.whitespace), '')))
    cleaned_description = cleaned_description.drop_duplicates()  # Remove duplicates
    return cleaned_description

def simple_tokenize(data):
    # Tokenize job descriptions by splitting by whitespace
    return data.apply(lambda s: [x.strip() for x in s])

def generate_ngrams(tokens, n):
    # Generate n-grams from tokenized data
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def create_ngram_index(cleaned_description):
    # Create dictionaries to store unigrams and bigrams
    unigram_index = {}
    bigram_index = {}

    for idx, token_list in enumerate(cleaned_description):
        # Generate unigrams
        unigrams = generate_ngrams(token_list, 1)
        for unigram in unigrams:
            if unigram not in unigram_index:
                unigram_index[unigram] = []
            unigram_index[unigram].append(idx)

        # Generate bigrams
        bigrams = generate_ngrams(token_list, 2)
        for bigram in bigrams:
            if bigram not in bigram_index:
                bigram_index[bigram] = []
            bigram_index[bigram].append(idx)

    return unigram_index, bigram_index

''' Indexing with set operation '''
def preprocess_and_stem_descriptions(cleaned_description):
    print('Preprocessing.')
    print('- Indexing..')
    # Replace non-alphabets with spaces, and collapse spaces
    cleaned_description = cleaned_description.apply(lambda s: re.sub(r'[^A-Za-z]', ' ', s))
    cleaned_description = cleaned_description.apply(lambda s: re.sub(r'\s+', ' ', s))

    print('- Tokenizing the descriptions..')
    # Tokenize job descriptions
    tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))

    print('- Removing stopwords..')
    # Remove stop words
    stop_dict = set(stopwords.words('english'))  # Specify the language
    sw_removed_description = tokenized_description.apply(lambda s: [word for word in OrderedSet(s) if word not in stop_dict])
    sw_removed_description = sw_removed_description.apply(lambda s: [word for word in s if len(word) > 2])  # Remove short words

    print('- Caching stem..')
    # Create stem caches for efficiency
    concated = np.unique(np.concatenate([s for s in sw_removed_description.values]))
    stem_cache = {}
    ps = PorterStemmer()
    for s in concated:
        stem_cache[s] = ps.stem(s)

    print('- Applying stem..')
    # Apply stemming
    stemmed_description = sw_removed_description.apply(lambda s: [stem_cache[w] for w in s])

    print('Preprocess done successfully !!')
    return stemmed_description

''' tf-idf scoring function '''
def apply_tf_idf_vectorizer(cleaned_description):
    print('Applying tf_idf vectorizer')
    # Initialize TF-IDF vectorizer
    print('- Initializing..')
    tf_idf_vectorizer = TfidfVectorizer(use_idf=True)
    # Fit and transform cleaned descriptions
    print('- Fitting..')
    tf_idf_vectorizer.fit(cleaned_description)
    transformed_data = tf_idf_vectorizer.transform(cleaned_description)
    
    # Convert to DataFrame
    print('- Converting to pandas dataframe..')
    X_tfidf_df = pd.DataFrame(transformed_data.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())
    print('Done TF-IDF vectorizing !!')
    return tf_idf_vectorizer, X_tfidf_df

''' BM25 '''
# page 87
class BM25(object):
    def __init__(self, vectorizer, b=0.75, k1=1.6):
        self.vectorizer = vectorizer
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        self.y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = self.y.sum(1).mean()

    def transform(self, q):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        len_y = self.y.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        y = self.y.tocsc()[:, q.indices]
        denom = y + (k1 * (1 - b + b * len_y / avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = y.multiply(np.broadcast_to(idf, y.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1

# Execution
if __name__ == "__main__":
    # Data file path
    file_path = "../Week 1/resource/software_developer_united_states_1971_20191023_1.csv"
    
    # Step 1: Preprocess Data
    cleaned_description = get_and_clean_data(file_path)
    stemmed_description = preprocess_and_stem_descriptions(cleaned_description)
    
    # Step 2: Create N-Gram Indices
    tokenized_data = simple_tokenize(stemmed_description)
    unigram_index, bigram_index = create_ngram_index(tokenized_data)
    
    # Step 3: Apply TF-IDF
    tf_idf_vectorizer, X_tfidf_df = apply_tf_idf_vectorizer(cleaned_description)
    
    # Step 4: Fit BM25
    bm25 = BM25(tf_idf_vectorizer)
    bm25.fit(cleaned_description)
    
    # Example Query
    query = 'cloud computing'
    scores = bm25.transform(query)
    ranked_indices = np.argsort(scores)[::-1]
    
    # Step 5: Combine with Unigram/Bigram Matches
    query_tokens = query.split()  # Tokenize the query
    unigram_matches = set()
    bigram_matches = set()
    
    # Check n-gram indices
    for token in query_tokens:
        unigram_matches.update(unigram_index.get((token,), []))
    bigram_query = generate_ngrams(query_tokens, 2)
    for bigram in bigram_query:
        bigram_matches.update(bigram_index.get(bigram, []))
    
    # Combine BM25 scores with n-gram matches
    final_results = pd.DataFrame({'BM25_Score': scores, 'Unigram_Match': 0, 'Bigram_Match': 0})
    final_results.loc[list(unigram_matches), 'Unigram_Match'] = 1
    final_results.loc[list(bigram_matches), 'Bigram_Match'] = 1
    
    # Sort and Display Top Results
    final_results['Combined_Score'] = final_results['BM25_Score'] + final_results['Unigram_Match'] + final_results['Bigram_Match']
    final_results = final_results.sort_values(by='Combined_Score', ascending=False)
    
    # Output
    print('---------------')
    print('Showing result')
    print(final_results.head())
