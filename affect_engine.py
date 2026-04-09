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

# def update_coord(prev, delta, persona, o, c, e, a, n, coord_type):
    # # 1. REFINED SIGNAL
    # raw_val = np.tanh(delta * 1.0) * 0.6 
    # normalized_delta = np.sign(raw_val) * (np.abs(raw_val) ** 1.1)
    
    # is_neurotic = n > 0.6
    # is_hostile = a < 0.3 and e > 0.6
    # is_avoidant = e < 0.3
    
    # # 2. INCREASED INERTIA (Lower Viscosity)
    # viscosity = 0.65 if (is_neurotic or is_hostile) else 0.40
    
    # offset = prev - persona
    
    # # 3. COORDINATE LOGIC WITH REDUCED GAIN
    # if coord_type == "A":
    #     stim_intensity = abs(normalized_delta)
    #     if is_neurotic: multiplier = 4.5 # Down from 6.0
    #     elif is_hostile: multiplier = 3.5 # Down from 4.5
    #     elif is_avoidant: multiplier = 0.8 
    #     else: multiplier = 1.2 
    #     final_change = (stim_intensity * multiplier)
    #     drift_speed = 0.02 if (is_neurotic or is_hostile) else 0.05

    # elif coord_type == "D":
    #     if is_hostile:
    #         multiplier = 0.8 
    #         final_change = abs(normalized_delta) * multiplier
    #     elif is_neurotic or is_avoidant:

    #         multiplier = 2.0 if delta < 0 else 0.4
    #         final_change = normalized_delta * multiplier
    #     else:
    #         multiplier = 0.8
    #         final_change = normalized_delta * multiplier
    #     drift_speed = 0.02 if (is_neurotic or is_hostile) else 0.05

    # elif coord_type == "P":
    #     # FIX: PLEASURE PLUNGE
    #     # Drastically reduced base sensitivity and personality multipliers
    #     p_sens = 1.2 + (n * 0.8) + ((1.0 - a) * 0.5) if delta < 0 else 1.0
    #     final_change = normalized_delta * p_sens
    #     # P-Drift is kept low to ensure the mood 'sticks'
    #     drift_speed = 0.01 if (is_neurotic or is_hostile) else 0.03
    
    # else:
    #     final_change = normalized_delta
    #     drift_speed = 0.02

    # # 4. STEP CAPPING (Critical for stopping the -1/+1 slam)
    # change = final_change * viscosity
    # max_step = 0.32
    # clamped_change = max(min(change, max_step), -max_step)
    
    # internal_drift = -offset * drift_speed
    
    # # Apply Damping for A and D to smooth out the jitter
    # impact = clamped_change
    # if coord_type != "P":
    #     damping = 1.0 / (1.0 + (0.4 * abs(offset)))
    #     impact *= damping
    
    # return max(min(prev + impact + internal_drift, 1.0), -1.0)

def update_coord(prev, delta, persona, o, c, e, a, n, coord_type):
    # 1. REFINED SIGNAL
    raw_val = np.tanh(delta * 1.0) * 0.6 
    normalized_delta = np.sign(raw_val) * (np.abs(raw_val) ** 1.1)
    
    is_neurotic = n > 0.6
    is_hostile = a < 0.3 and e > 0.6
    is_avoidant = e < 0.3
    
    # 2. INCREASED INERTIA (Lower Viscosity)
    viscosity = 0.8 if (is_neurotic or is_hostile) else 0.6
    
    offset = prev - persona
    
    # 3. COORDINATE LOGIC WITH REDUCED GAIN
    if coord_type == "A":
        stim_intensity = abs(normalized_delta)
        if is_neurotic: multiplier = 4.5 # Down from 6.0
        elif is_hostile: multiplier = 3.5 # Down from 4.5
        elif is_avoidant: multiplier = 0.8 
        else: multiplier = 1.2 
        final_change = (stim_intensity * multiplier)
        drift_speed = 0.02 if (is_neurotic or is_hostile) else 0.05

    elif coord_type == "D":
        if is_hostile:
            multiplier = 0.8 
            final_change = abs(normalized_delta) * multiplier
        elif is_neurotic or is_avoidant:

            multiplier = 2.0 if delta < 0 else 0.4
            final_change = normalized_delta * multiplier
        else:
            multiplier = 0.8
            final_change = normalized_delta * multiplier
        drift_speed = 0.02 if (is_neurotic or is_hostile) else 0.05

    elif coord_type == "P":
        # FIX: PLEASURE PLUNGE
        # Drastically reduced base sensitivity and personality multipliers
        p_sens = 1.2 + (n * 0.8) + ((1.0 - a) * 0.5) if delta < 0 else 1.0
        final_change = normalized_delta * p_sens
        # P-Drift is kept low to ensure the mood 'sticks'
        drift_speed = 0.01 if (is_neurotic or is_hostile) else 0.03
    
    else:
        final_change = normalized_delta
        drift_speed = 0.02

    # 4. STEP CAPPING (Critical for stopping the -1/+1 slam)
    change = final_change * viscosity
    max_step = 0.32
    clamped_change = max(min(change, max_step), -max_step)
    
    internal_drift = -offset * drift_speed
    
    # Apply Damping for A and D to smooth out the jitter
    impact = clamped_change
    if coord_type != "P":
        damping = 1.0 / (1.0 + (0.4 * abs(offset)))
        impact *= damping
    
    return max(min(prev + impact + internal_drift, 1.0), -1.0)