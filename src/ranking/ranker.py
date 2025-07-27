def rank_sections(similarities):
    """Assign importance_rank based on descending similarity."""
    ranked = sorted(similarities, key=lambda x: -x[1])
    for rank, (meta, score) in enumerate(ranked, start=1):
        meta['importance_rank'] = rank
        meta['similarity'] = float(score)
    return ranked
