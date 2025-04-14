import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_scores(object_names, brand_names):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(object_names + brand_names)

    cosine_similarities = cosine_similarity(
        tfidf_matrix[:len(object_names)],
        tfidf_matrix[len(object_names):]
    )

    results = []
    for i, object_name in enumerate(object_names):
        for j, brand_name in enumerate(brand_names):
            cosine_sim = cosine_similarities[i, j]
            fuzz_score = fuzz.partial_ratio(object_name, brand_name)
            combined_similarity_score = (cosine_sim + (fuzz_score / 100.0)) / 2

            results.append({
                'object_name': object_name,
                'brand_name': brand_name,
                'cosine_similarity': cosine_sim,
                'fuzz_partial_ratio': fuzz_score,
                'combined_similarity_score': combined_similarity_score
            })

    return pd.DataFrame(results)

# Sample input lists
object_names = ["Nike Air Max 270", "Adidas Ultraboost", "Puma Running Shoes", "Reebok Classic"]
brand_names = ["Nike", "Adidas", "Puma", "Reebok"]

# Compute and filter matches
similarity_df = compute_similarity_scores(object_names, brand_names)
filtered_df = similarity_df[similarity_df["combined_similarity_score"] > 0.5]

# Output
print(filtered_df)
