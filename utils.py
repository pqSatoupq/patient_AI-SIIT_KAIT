import re, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import datetime, json, os
from config import EMOTION_MAP

def build_dashboard(p, a, d, prev_p, prev_a, prev_d, label):
    dp, da, dd = p - prev_p, a - prev_a, d - prev_d
    def get_trend(delta):
        if abs(delta) < 0.005: return "<span style='color:gray; font-size:0.8em;'> (stable)</span>"
        color = "#00ff88" if delta > 0 else "#ff4444"
        arrow = "↑" if delta > 0 else "↓"
        return f"<span style='color:{color}; font-weight:bold;'> {arrow} {abs(delta):.2f}</span>"
    pw, aw, dw = (p+1)*50, (a+1)*50, (d+1)*50
    return f"""
    <div style='background: #121212; padding: 20px; border-radius: 12px; color: white; border: 1px solid #333;'>
        <h3 style='margin:0 0 15px 0; color: #4facfe;'>Current Mood: {label}</h3>
        <div style='margin-bottom:12px;'><strong>Pleasure (Hope):</strong> {p:.2f} {get_trend(dp)}<div style='background:#333; width:100%; height:8px;'><div style='background:linear-gradient(90deg, #ff4b2b, #00ff88); width:{pw}%; height:8px;'></div></div></div>
        <div style='margin-bottom:12px;'><strong>Arousal (Stress):</strong> {a:.2f} {get_trend(da)}<div style='background:#333; width:100%; height:8px;'><div style='background:linear-gradient(90deg, #3a7bd5, #ff4444); width:{aw}%; height:8px;'></div></div></div>
        <div><strong>Dominance (Control):</strong> {d:.2f} {get_trend(dd)}<div style='background:#333; width:100%; height:8px;'><div style='background:linear-gradient(90deg, #4b6cb7, #182848); width:{dw}%; height:8px;'></div></div></div>
    </div>
    """

def log_mood_journey(history_list, session_id):
    if not history_list: return None
    df = pd.DataFrame(history_list, columns=['Turn', 'Pleasure', 'Arousal', 'Dominance', 'Emotion'])
    filename = f"mood_log_{session_id}.csv"
    df.to_csv(filename, index=False)
    return filename

def update_plot(history_data):
    if not history_data or len(history_data) == 0: return None
    df = pd.DataFrame(history_data, columns=['Turn', 'P', 'A', 'D', 'Emotion'])
    fig = plt.figure(figsize=(5, 4))
    plt.plot(df['Turn'], df['P'], label='Pleasure', marker='o', color='#00ff88', linewidth=2)
    plt.plot(df['Turn'], df['A'], label='Arousal', marker='o', color='#ff4444', linewidth=2)
    plt.plot(df['Turn'], df['D'], label='Dominance', marker='o', color='#3a7bd5', linewidth=2)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color='white', linewidth=0.5, alpha=0.3)
    plt.title("Patient Emotional Trajectory", color='white')
    fig.patch.set_facecolor('#121212')
    plt.gca().set_facecolor('#1e1e1e')
    plt.gca().tick_params(colors='white')
    plt.legend(loc='upper right', facecolor='#121212', labelcolor='white')
    plt.tight_layout()
    return fig

def export_session_json(history, mood_history, o, c, e, a, n, situation):
    data = {
        "metadata": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": situation,
            "patient_traits": {"O": o, "C": c, "E": e, "A": a, "N": n}
        },
        "turns": []
    }
    doctor_turns = [m["content"] for m in history if m["role"] == "user"]
    patient_turns = [m["content"] for m in history if m["role"] == "assistant"]
    for i in range(len(patient_turns)):
        data["turns"].append({
            "turn_number": i + 1,
            "doctor_input": doctor_turns[i] if i < len(doctor_turns) else "",
            "patient_response": patient_turns[i],
            "pad_metrics": {
                "P": mood_history[i][1], "A": mood_history[i][2], "D": mood_history[i][3], "label": mood_history[i][4]
            }
        })
    file_path = "medical_teacher_report.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return file_path

import os

def load_all_scenarios():
    scenarios = {}
    folder = "scenarios"
    if not os.path.exists(folder):
        os.makedirs(folder)
        return {}
        
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                    content = f.read()
                    
                    # FIX: Only split on tags that ARE NOT followed by a colon.
                    # This prevents splitting on [PATIENT RESPONSE]: inside the instructions.
                    chunks = re.split(r'^\s*\[([A-Z\s_]+)\](?!\s*:)', content, flags=re.MULTILINE)
                    
                    parts = {}
                    # The split returns [text_before, tag_name, content, tag_name, content...]
                    for i in range(1, len(chunks), 2):
                        tag_name = chunks[i].strip()
                        tag_content = chunks[i+1].strip()
                        parts[tag_name] = tag_content
                    
                    scenarios[filename.replace(".txt", "")] = parts
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return scenarios

def validate_patient_output(raw_text):
    """
    Enhanced validation: Checks for missing tags AND detects 
    if tags are repeated (looping errors).
    """
    required_tags = [
        'LINGUISTIC ANALYSIS', 
        'INTERNAL THOUGHT', 
        'EMOTIONAL STATE', 
        'PATIENT RESPONSE'
    ]
    
    for tag in required_tags:
        # Use findall to count every instance of the tag
        matches = re.findall(rf"\[{tag}\]\s*:", raw_text, re.IGNORECASE)
        
        # ERROR 1: Tag is missing
        if len(matches) == 0:
            print(f"Validation Error: Missing tag [{tag}]")
            return False
            
        # ERROR 2: Tag is repeated (The LLM is looping)
        if len(matches) > 1:
            print(f"Validation Error: Repeated tag [{tag}] ({len(matches)} times)")
            return False
    
    # Final check: Ensure the response isn't empty
    response_match = re.search(r"\[PATIENT RESPONSE\]\s*:\s*(.*)", raw_text, re.DOTALL | re.IGNORECASE)
    if not response_match or len(response_match.group(1).strip()) < 5:
        return False
        
    return True

