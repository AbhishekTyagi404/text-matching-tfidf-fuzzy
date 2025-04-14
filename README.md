# Brand Similarity Matching using TF-IDF + Fuzzy Logic

This repository contains a Python script for matching free-text cost object descriptions (such as internal orders or line items) with standardized brand names using a hybrid similarity approach.

Developed by [Abhishek Tyagi](https://github.com/AbhishekTyagi404), this solution combines:
- **TF-IDF Vectorization + Cosine Similarity**: Captures semantic closeness.
- **Fuzzy String Matching (Levenshtein Distance)**: Captures textual resemblance (typos, truncations, etc.)
- **Combined Score**: A weighted average of both for robust matching.

## 🔍 Use Case

This is especially useful in procurement analytics, vendor data unification, or any domain where data entry inconsistencies lead to non-standard brand references.

---

## 🧠 How It Works

1. **Load object names and brand names** as Python lists (can be replaced with real datasets).
2. **Vectorize the text** using `TfidfVectorizer`.
3. **Compute cosine similarity** between each pair of object and brand names.
4. **Apply fuzzy matching** using `fuzz.partial_ratio`.
5. **Calculate a final combined similarity score**.
6. **Filter based on a threshold** to find likely brand mappings.

---

## 📦 Dependencies

- Python 3.7+
- `pandas`
- `scikit-learn`
- `fuzzywuzzy`
- `python-Levenshtein` (optional but recommended for performance)

Install dependencies:
```bash
pip install pandas scikit-learn fuzzywuzzy python-Levenshtein
```

---

## 🧪 Example

```bash
python brand_similarity_full.py
```

Example output:
```
Top Brand Matches Based on Combined Score:
           object_name brand_name  cosine_similarity  fuzz_partial_ratio  combined_similarity_score
0     Nike Air Max 270       Nike           0.432...                  86                     0.646
1  Adidas Ultraboost     Adidas           0.391...                  90                     0.645
...
```

---

## 📁 Files

- `brand_similarity_full.py` - Full Python script with docstrings and example.
- `brand_similarity_tf_idf.py` - Minimal version for quick deployment.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributions

Pull requests and stars are welcome. If you find it helpful, consider citing or referencing this work.

---

## 🌐 Author

Abhishek Tyagi  
[GitHub](https://github.com/AbhishekTyagi404) • [Website](https://kritrimintelligence.com)

