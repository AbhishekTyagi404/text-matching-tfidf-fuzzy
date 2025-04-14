"""
Brand Matching using TF-IDF and Fuzzy Logic

This script demonstrates a hybrid approach to match cost object descriptions
to brand names using TF-IDF cosine similarity combined with fuzzy string matching
(Levenshtein distance). Designed to work at scale using pandas + scikit-learn.

Author: Abhishek Tyagi
License: MIT
"""

import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_scores(object_names, brand_names):
    """
    Compute pairwise similarity scores between object names and brand names.
    Uses a combination of TF-IDF cosine similarity and FuzzyWuzzy partial ratio.
    
    Args:
        object_names (list of str): List of free-text object descriptions.
        brand_names (list of str): List of known brand names.

    Returns:
        pd.DataFrame: A DataFrame with object_name, brand_name, cosine similarity,
                      fuzzy score, and the combined score.
    """
    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(object_names + brand_names)

    # Cosine similarity matrix (object_name rows vs brand_name columns)
    cosine_similarities = cosine_similarity(
        tfidf_matrix[:len(object_names)],
        tfidf_matrix[len(object_names):]
    )

    # Compute combined similarity
    results = []
    for i, object_name in enumerate(object_names):
        for j, brand_name in enumerate(brand_names):
            cosine_sim = cosine_similarities[i, j]
            fuzz_score = fuzz.partial_ratio(object_name, brand_name)
            combined_score = (cosine_sim + (fuzz_score / 100.0)) / 2

            results.append({
                'object_name': object_name,
                'brand_name': brand_name,
                'cosine_similarity': cosine_sim,
                'fuzz_partial_ratio': fuzz_score,
                'combined_similarity_score': combined_score
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    object_names = [
        "Nike Air Max 270", 
        "Adidas Ultraboost", 
        "Puma Running Shoes", 
        "Reebok Classic"
    ]
    brand_names = ["Nike", "Adidas", "Puma", "Reebok"]

    # Run similarity scoring
    df_similarity = compute_similarity_scores(object_names, brand_names)

    # Filter based on a practical threshold for matching
    threshold = 0.5
    matched_df = df_similarity[df_similarity["combined_similarity_score"] >= threshold]

    # Output result
    print("Top Brand Matches Based on Combined Score:")
    print(matched_df.sort_values("combined_similarity_score", ascending=False))
