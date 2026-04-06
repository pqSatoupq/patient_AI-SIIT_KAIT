import numpy as np
from config import EMOTION_MAP


def get_mixed_labels(p, a, d):
    current = np.array([p, a, d])
    distances = {name: np.linalg.norm(current - np.array(val)) for name, val in EMOTION_MAP.items()}
    sorted_emotions = sorted(distances.items(), key=lambda x: x[1])
    closest_name, closest_dist = sorted_emotions[0]
    dynamic_threshold = closest_dist * 0.15 
    top_labels = [name for name, dist in sorted_emotions if dist <= closest_dist + dynamic_threshold]
    return " / ".join(top_labels[:2])

def update_coord(prev, delta, persona, o, c, e, a, n, coord_type):
    # This logic implements Affective Chronometry (Davidson, 1998) 
    # and Cognitive Appraisal (Lazarus, 1991)
    # normalized_delta = np.tanh(delta * 1.1) * 0.65 
    normalized_delta = np.tanh(delta * 1.1) * 0.35
    # viscosity = 0.9 if n < 0.3 else (0.4 if n > 0.6 else 0.65)
    viscosity = 0.85 if n < 0.3 else (0.55 if n > 0.6 else 0.7)
    
    multiplier = 1.0
    offset = prev - persona
    n_power = np.power(n, 1.8) 
    
    if coord_type == "D":
        if delta > 0.5: normalized_delta = abs(normalized_delta) * 0.2 
        if n > 0.6 or e < 0.3:
            normalized_delta = -abs(normalized_delta) if abs(delta) > 0.05 else -0.05
            multiplier = 2.5 * n_power
        else:
            multiplier = 0.2
            
    elif coord_type == "A":
        if abs(delta) > 0.35 and n < 0.3 and a < 0.2: 
            multiplier = 2.2 
        elif n > 0.6:
            if delta < 0: normalized_delta = abs(normalized_delta) * 1.6
            multiplier = 2.0
        else:
            multiplier = 0.35

    elif coord_type == "P":
        if delta < 0: multiplier = 1.0 + (1.2 * np.power(n, 2.0))

    is_recovering = (normalized_delta > 0 and prev < persona) or (normalized_delta < 0 and prev > persona)
    rumination = 3.0 * n 
    damping = 1.0 / (1.0 + (rumination * abs(offset))) if is_recovering else 1.0 / (1.0 + (0.3 * abs(offset)))
    drift_speed = 0.12 if n < 0.3 else (0.02 if n > 0.6 else 0.06)

    resilience_factor = (1.0 - n) + (e * 0.05) 

    raw_impact = normalized_delta * multiplier * damping * viscosity
    word_impact = max(min(raw_impact, 0.45), -0.45)
    # word_impact = normalized_delta * multiplier * damping * viscosity
    internal_drift = -offset * (drift_speed * resilience_factor)
    
    return max(min(prev + word_impact + internal_drift, 1.0), -1.0)