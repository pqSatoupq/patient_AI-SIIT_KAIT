# Use local relative paths for Docker compatibility
LLAMA_PATH = "./models/Llama3.2-3B"
DEBERTA_PATH = "./models/deberta-pad"
PROJECTOR_WEIGHTS = "./models/llama_projector_aligned2.pt"
DEVICE = "cuda"

# EMOTION_MAP = {
#     "Anger": [-0.51, 0.59, 0.25], "Disappointment": [-0.3, 0.1, -0.4],
#     "Anxious": [-0.64, 0.60, -0.43], "Gratitude": [0.4, 0.2, -0.3],
#     "Satisfaction": [0.3, -0.2, 0.4], "Denial": [-0.1, 0.0, 0.6],
#     "Confusion": [-0.2, 0.4, -0.5], "Sadness": [-0.6, -0.3, -0.4],
#     "Despair": [-0.9, -0.4, -0.8]
# }

EMOTION_MAP = {
    "Anger": [-0.51, 0.59, 0.25], 
    "Disappointment": [-0.3, 0.1, -0.4],
    "Anxious": [-0.64, 0.60, -0.43], 
    "Gratitude": [0.4, 0.2, -0.3],
    "Sadness": [-0.6, -0.3, -0.4],
    "Despair": [-0.9, -0.4, -0.8]
}

PRESETS = {
    "Alex (Anxious)": [0.5, 0.5, 0.4, 0.4, 0.8],
    "Alex (Hostile)": [0.4, 0.5, 0.8, 0.1, 0.4],
    "Alex (Resilient)": [0.7, 0.6, 0.6, 0.8, 0.3],
    "Alex (Calm)": [0.5, 0.8, 0.4, 0.7, 0.2]
}