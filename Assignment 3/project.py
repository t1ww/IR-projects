# Imports
import os
import re
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from ordered_set import OrderedSet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


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
    print('Preprocessing...')
    print('Indexing...')
    # Replace non-alphabets with spaces, and collapse spaces
    cleaned_description = cleaned_description.apply(lambda s: re.sub(r'[^A-Za-z]', ' ', s))
    cleaned_description = cleaned_description.apply(lambda s: re.sub(r'\s+', ' ', s))

    print('Tokenizing the descriptions...')
    # Tokenize job descriptions
    tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))

    print('Removing stopwords...')
    # Remove stop words
    stop_dict = set(stopwords.words('english'))  # Specify the language
    sw_removed_description = tokenized_description.apply(lambda s: [word for word in OrderedSet(s) if word not in stop_dict])
    sw_removed_description = sw_removed_description.apply(lambda s: [word for word in s if len(word) > 2])  # Remove short words

    print('Caching stem...')
    # Create stem caches for efficiency
    concated = np.unique(np.concatenate([s for s in sw_removed_description.values]))
    stem_cache = {}
    ps = PorterStemmer()
    for s in concated:
        stem_cache[s] = ps.stem(s)

    print('Applying stem...')
    # Apply stemming
    stemmed_description = sw_removed_description.apply(lambda s: [stem_cache[w] for w in s])

    print('Preprocess successfully !!')
    return stemmed_description

# Execution
if __name__ == "__main__":
    # Data file path
    print(os.getcwd())
    file_path = "../Week 1/resource/software_developer_united_states_1971_20191023_1.csv"
    
    # Preprocess
    cleaned_description = get_and_clean_data(file_path)
    stemmed_description = preprocess_and_stem_descriptions(cleaned_description)
    
    # Create indices
    tokenized_data = simple_tokenize(stemmed_description)
    unigram_index, bigram_index = create_ngram_index(tokenized_data)