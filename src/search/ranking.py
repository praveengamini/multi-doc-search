def explain_match(query, document):
    query_tokens = set(query.split())
    doc_tokens = set(document.split())

    overlap = query_tokens.intersection(doc_tokens)
    ratio = len(overlap) / max(len(query_tokens), 1)

    return {
        "overlapping_keywords": list(overlap),
        "overlap_ratio": ratio
    }
