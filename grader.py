def grade(score):
    """
    Normalize reward score to 0.0 - 1.0 range
    """
    max_score = 30   # expected good performance
    min_score = 0

    normalized = (score - min_score) / (max_score - min_score)

    # clamp between 0 and 1
    return max(0.01, min(0.99, normalized))
