## Create a simple command-line-interface application that search the job description data (Given)

- Use a Python library like argparse to create a command-line interface application that accepts a search query. (1 point)
- Preprocess the text data to create 1-gram and 2-grams (unigram and bigrams) for each document. (0.25 point each = 0.5 point)
- Index these grams for efficient searching. (0.25 point each = 0.5 point)
- Implement tf-idf, and bm25 scoring functions. (0.25 point each = 0.5 point)
- For a query input by a user, calculate the scores between it and all the documents. Then, sort all the product and display the top 5 ranks for each scoring mechanism. (0.5 point each = 1 points)
- Provide an in-depth analysis of how bm25 scoring may provide better results in certain contexts due to its sophisticated scoring mechanism.
    - With an example query in which bm25 perform better (0.25 point)
    - Based on the example query, discuss the case bm25 will perform (0.5 point)
    - With an example query in which bm25 perform worse (0.25 point)
    - Based on the example query, discuss the case bm25 will be less likely to perform (0.5 point)

---
Key planning
- Makes cli that takes search query input
- Preprocess the text data to create 1-gram and 2-grams (unigram and bigrams) for each document.
- Index these grams for efficient searching.
- Implement tf-idf, and bm25 scoring functions.
- For a query input by a user, calculate the scores between it and all the documents. Then, sort all the product and display the top 5 ranks for each scoring mechanism.
- Provide an in-depth analysis of how bm25 scoring may provide better results in certain contexts due to its sophisticated scoring mechanism.
    - See above
---
# Task Listing
- Job Search CLI Application Tasks

## **1. Setting Up the CLI**
- [ ] Install and configure the `argparse` library for the command-line interface.
- [ ] Create a CLI application that accepts a search query as input.

---

## **2. Preprocessing Text Data**
- [ ] Load and clean the job description data.
- [ ] Create **1-grams (unigrams)** for each document.
- [ ] Create **2-grams (bigrams)** for each document.
- [ ] Store these grams in a structure suitable for efficient searching (e.g., dictionaries or inverted indices).

---

## **3. Indexing for Efficient Search**
- [ ] Index the unigrams.
- [ ] Index the bigrams.
- [ ] Ensure the indices allow for efficient retrieval and scoring.

---

## **4. Implementing Scoring Mechanisms**
- [ ] Implement the **TF-IDF scoring function**.
- [ ] Implement the **BM25 scoring function**.

---

## **5. Query Processing and Ranking**
- [ ] Accept a query input from the user through the CLI.
- [ ] Preprocess the query to generate unigrams and bigrams.
- [ ] Calculate scores for the query against all documents using **TF-IDF**.
- [ ] Calculate scores for the query against all documents using **BM25**.
- [ ] Sort and display the **top 5 ranked documents** for each scoring mechanism.

---

## **6. In-depth Analysis of BM25**
- [ ] **Example Query Where BM25 Performs Better**:
  - [ ] Provide an example query where BM25 gives better results.
  - [ ] Explain why BM25 performed better in this case (e.g., considering term frequency saturation, document length normalization).
- [ ] **Example Query Where BM25 Performs Worse**:
  - [ ] Provide an example query where BM25 gives worse results.
  - [ ] Explain why BM25 performed worse in this case (e.g., due to short query terms or overly normalized results).

---

## **Optional: Testing and Validation**
- [ ] Write test cases to validate preprocessing (unigram and bigram creation).
- [ ] Test the accuracy of the TF-IDF and BM25 implementation.
- [ ] Validate the sorting and ranking outputs for both scoring mechanisms.