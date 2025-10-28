def to_label(x: float) -> str:
    x = max(-1.0, min(1.0, float(x)))
    if x < -0.6: return "extremely negative"
    if x < -0.2: return "negative"
    if x <  0.2: return "neutral"
    if x <  0.6: return "positive"
    return "extremely positive"
