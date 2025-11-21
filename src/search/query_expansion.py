# src/search/query_expansion.py
import nltk
from nltk.corpus import wordnet as wn
from src.preprocess.cleaner import clean_text

# Ensure NLTK wordnet is available (download outside of runtime if possible)
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def get_synonyms(token):
    syns = set()
    for syn in wn.synsets(token):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != token.lower():
                syns.add(name.lower())
    return list(syns)

def expand_query_text(query, max_terms_per_token=2):
    """
    Expand query by looking up synonyms for tokens.
    Returns expanded string containing original query + synonyms.
    """
    q = clean_text(query)
    tokens = nltk.word_tokenize(q)
    expanded_terms = []
    for t in tokens:
        # Skip punctuation/numbers
        if not t.isalpha():
            continue
        syns = get_synonyms(t)[:max_terms_per_token]
        expanded_terms.extend(syns)
    # combine: original query + synonyms
    expanded = q + " " + " ".join(expanded_terms)
    return expanded
