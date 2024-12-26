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