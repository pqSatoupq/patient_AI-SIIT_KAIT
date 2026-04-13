import re, torch, gradio as gr, pandas as pd, numpy as np
import datetime, os
from config import *
# Import the local models directly
from inference import de_tokenizer, de_model, llama_model, llama_tokenizer, projector
from affect_engine import update_coord, get_mixed_labels
from utils import build_dashboard, log_mood_journey, update_plot, export_session_json, load_all_scenarios, validate_patient_output, get_face_image, AVATAR_MAP

# Set the default character for the Gradio Dropdown
default_char = list(AVATAR_MAP.keys())[0] if AVATAR_MAP else "None"

# --- NEW: AFFECTIVE STEERING HOOK CLASS ---
class AffectiveSteeringHook:
    """
    Implements Additive Steering by nudging the model's hidden states 
    toward the PAD vector without changing the sequence length.
    """
    def __init__(self, soft_prompt, alpha=1.0):
        self.soft_p = soft_prompt # Shape: (1, 1, 3072)
        self.alpha = alpha

    def __call__(self, module, input, output):
        # Transformer layers return a tuple (hidden_states, attention_metadata)
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Additive 'nudge' across all tokens in the current sequence
            # Broadcasting handles the (1, seq_len, 3072) vs (1, 1, 3072) math
            new_hidden = hidden_states + (self.alpha * self.soft_p)
            return (new_hidden,) + output[1:]
        else:
            # Standard embedding layer returns a single Tensor
            return output + (self.alpha * self.soft_p)

def get_tag(tag, text):
    pattern = rf"\[?{tag}\]?\s*[:\-]?\s*(.*?)(?=\n\s*\[|\n\s*[A-Z\s]+:|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        content = re.sub(r"^[,\s\:]+", "", content)
        return content
    return ""

def generate_commentary(history, mood_history, situation):
    if not history:
        return [{"role": "assistant", "content": "⚠️ No conversation data available for analysis."}]

    transcript = ""
    for msg in history:
        role = "Doctor" if msg["role"] == "user" else "Patient"
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
        
        if role == "Patient":
            match = re.search(r"\[PATIENT RESPONSE\]:\s*(.*)", str(content), re.DOTALL)
            content = match.group(1).strip() if match else content
        transcript += f"{role}: {content}\n\n"

    supervisor_sys = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
ROLE: Expert Medical Educator & Senior Clinical Supervisor.
CONTEXT: A medical student (Doctor) is interacting with a simulated patient (Alex) in a high-stakes scenario.
SCENARIO: {situation}

TASK:
1. Analyze the student's(Doctor) adherence to the SPIKES protocol (Setting, Perception, Invitation, Knowledge, Empathy, Strategy).
2. Correlate the student's words with the P (Pleasure) and A (Arousal) shifts. Final State: {mood_history[-1][4] if mood_history else 'N/A'}.
3. Successes vs. Distress: Identify where the student succeeded and where they caused unnecessary patient distress (evidenced by spikes in Arousal).
4. Evidence: Use specific quotes from the transcript below to justify your feedback.
5. Grading: give the score (0-10) for each aspect for feedback.
6. advicing comment that what is medical student is doing good and what their need to needs improvement.

Provide a 'CLINICAL FEEDBACK REPORT' that is professional, critical yet constructive.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
TRANSCRIPT:
{transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

[CLINICAL FEEDBACK REPORT]:"""

    inputs = llama_tokenizer(supervisor_sys, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        out_ids = llama_model.generate(
            **inputs, 
            max_new_tokens=1024, 
            temperature=0.6, 
            do_sample=True,
            pad_token_id=llama_tokenizer.pad_token_id
        )
    
    raw_feedback = llama_tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return [{"role": "assistant", "content": f"### 🩺 CLINICAL SUPERVISOR DEBRIEF\n\n{raw_feedback}"}]

def patient_respond(message, history, o, c, e, a, n, sys_p, temp, tokens, penalty, intensity, pad_state, mood_history, situation, target_layer, target_layer_2, dual_mode, dist_mode, emo_mode, avatar_char):
    if not message: return history, gr.update(), "", pad_state, gr.update(), mood_history, None
    prev_p, prev_a, prev_d = pad_state

    # Calculate persona baseline [cite: 98]
    po, ao, do = (0.21*e + 0.59*a + 0.19*n), (0.15*o + 0.3*a - 0.57*n), (0.25*o + 0.17*c + 0.6*e - 0.32*a)

    label = get_mixed_labels(prev_p, prev_a, prev_d)

    with torch.inference_mode():

        # [1] Perception & Math 
        """*****************************************"""
        context_input = f"[SCENARIO]: {situation}. [CURRENT STATE]: {label}. [DOCTOR]: {message}"
        de_in = de_tokenizer(context_input, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        """*****************************************"""
        # de_in = de_tokenizer(message, return_tensors="pt").to(DEVICE)
        raw_d = de_model(de_in.input_ids, de_in.attention_mask) 
        dp, da, dd = raw_d[0].tolist()

        np_val = update_coord(prev_p, dp, po, o, c, e, a, n, "P")
        na_val = update_coord(prev_a, da, ao, o, c, e, a, n, "A")
        nd_val = update_coord(prev_d, dd, do, o, c, e, a, n, "D")
        new_pad = [np_val, na_val, nd_val]
        label = get_mixed_labels(np_val, na_val, nd_val)

        # [2] Prompt Construction
        clean_history = ""
        for msg in history[-4:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # --- ADD THIS FIX HERE ---
            if isinstance(content, list):
                # Flattens Gradio 5 list of dicts into a single string
                content = " ".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
            # -------------------------
            if role == "assistant":
                match = re.search(r"\[PATIENT RESPONSE\]:\s*(.*)", content, re.DOTALL)
                content = match.group(1).strip() if match else content
            clean_history += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{sys_p}
[CURRENT PAD: P:{np_val:.2f}, A:{na_val:.2f}, D:{nd_val:.2f}]
[EMOTIONAL BLEND: {label}]
(Context: {situation})
INSTRUCTION: Follow the pattern: [LINGUISTIC ANALYSIS], [INTERNAL THOUGHT], then [PATIENT RESPONSE].<|eot_id|>
{clean_history}<|start_header_id|>user<|end_header_id|>

Doctor: {message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

[LINGUISTIC ANALYSIS]:"""
    max_retries = 3
    attempts = 0
    is_valid = False
    full_new_text = ""
    current_temp = float(temp) # Use a local variable to tweak temperature if needed

    while attempts < max_retries and not is_valid:
        attempts += 1
        
        with torch.inference_mode():
            # [3] HOOK REGISTRATION
            soft_p = projector(torch.tensor([new_pad], device=DEVICE, dtype=torch.bfloat16))
            soft_p = torch.nn.functional.normalize(soft_p, p=2, dim=-1)
            
            handles = []
            if dual_mode:
                # Use set() just in case the user accidentally sets both sliders to the same number
                layers_to_hook = list(set([target_layer, target_layer_2]))
                # Split the intensity evenly across the two layers so we don't blow up the VRAM
                alpha = intensity / len(layers_to_hook)
            elif dist_mode:
                layers_to_hook = range(max(0, target_layer-2), min(32, target_layer+3))
                alpha = intensity / 10.0
            else:
                layers_to_hook = [target_layer]
                alpha = intensity / 2.0

            for i in layers_to_hook:
                module = llama_model.get_input_embeddings() if i == 0 else llama_model.model.layers[i-1]
                hook = AffectiveSteeringHook(soft_p, alpha=alpha)
                handles.append(module.register_forward_hook(hook))

            # [4] Generation
            inputs = llama_tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
            out_ids = llama_model.generate(
                **inputs,
                max_new_tokens=int(tokens), 
                temperature=current_temp, 
                repetition_penalty=float(penalty), 
                do_sample=True, 
                pad_token_id=llama_tokenizer.pad_token_id
            )

            # [5] HOOK CLEANUP (Must happen before we decide to retry or finish)
            for h in handles: h.remove()

        # [6] Validation Check
        raw_res = llama_tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        raw_res = re.sub(r"^[,\s\:]+", "", raw_res)
        temp_text = "[LINGUISTIC ANALYSIS]: " + raw_res
        
        # Call the validator from utils.py
        is_valid = validate_patient_output(temp_text)
        
        if is_valid:
            full_new_text = temp_text
        else:
            print(f"⚠️ Attempt {attempts} failed validation (tags missing or repeated).")
            current_temp = min(current_temp + 0.15, 1.2) # Increase randomness to "break" the loop

    # [7] Final Fallback if all retries fail
    if not is_valid:
        full_new_text = (
            "[LINGUISTIC ANALYSIS]: Patient is severely dissociated.\n"
            "[INTERNAL THOUGHT]: I can't find the words. Everything is blurry.\n"
            f"[EMOTIONAL STATE]: {label}\n"
            "[PATIENT RESPONSE]: I... I'm sorry. I can't focus. What was the question?"
        )

    # [8] Final Parsing (Now using full_new_text)
    analysis = get_tag('LINGUISTIC ANALYSIS', full_new_text)
    raw_thought = get_tag('INTERNAL THOUGHT', full_new_text)
    ai_label = get_tag('EMOTIONAL STATE', full_new_text)
    response = get_tag('PATIENT RESPONSE', full_new_text)

    if not response or len(response.strip()) < 2:
        response = "I... I can't think. Everything feels so loud. What should i do?"

    if emo_mode:
        display_label = label
        current_face = get_face_image(avatar_char, display_label)
    else:
        display_label = ai_label if (ai_label and len(ai_label) > 2) else label
        label_char = label
        current_face = get_face_image(avatar_char, label_char)
    
    final_msg = (
        f"[LINGUISTIC ANALYSIS]: {analysis}\n"
        f"[INTERNAL THOUGHT]: {raw_thought} (ΔP:{dp:.2f}, ΔA:{da:.2f}, ΔD:{dd:.2f})\n"
        f"[EMOTIONAL STATE]: {display_label} (P:{np_val:.2f}, A:{na_val:.2f}, D:{nd_val:.2f})\n"
        f"[PATIENT RESPONSE]: {response}"
    )
    
    history.append({"role": "user", "content": f"Doctor: {message}"})
    history.append({"role": "assistant", "content": final_msg})
    
    mood_history.append([len(mood_history) + 1, np_val, na_val, nd_val, display_label])
    session_id = datetime.datetime.now().strftime("%Y%m%d")
    csv_path = log_mood_journey(mood_history, session_id)
    current_plot = update_plot(mood_history)

    torch.cuda.empty_cache()
    return history, build_dashboard(np_val, na_val, nd_val, prev_p, prev_a, prev_d, display_label), "", new_pad, current_plot, mood_history, csv_path, current_face

# --- UI LAYOUT ---
MAX_CHARS = 500
# with gr.Blocks(theme=gr.themes.Soft()) as demo:
with gr.Blocks() as demo:
    all_scenarios = load_all_scenarios()
    pad_state = gr.State([-0.1, 0.0, 0.6])
    mood_history_state = gr.State([])

    with gr.Column(visible=False, variant="panel") as scenario_panel:
        gr.Markdown("## 📂 Select Clinical Scenario")
        scenario_drop = gr.Dropdown(label="Available Scenarios", choices=list(all_scenarios.keys()))
        scenario_desc = gr.Textbox(label="Description", interactive=False, lines=2)
        with gr.Row():
            btn_confirm_scen = gr.Button("✅ Confirm", variant="primary")
            btn_cancel_scen = gr.Button("❌ Cancel")

    gr.Markdown("# 🩺 Clinical Patient AI v3.6 (Forward Hook Edition)")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🛠️ Persona & Steering")
                btn_open_scen = gr.Button("📂 Select Scenario", variant="secondary")
                start_emo_dd = gr.Dropdown(label="Starting Emotion", choices=list(EMOTION_MAP.keys()), value="Gratitude")
                preset_dd = gr.Dropdown(label="Personality Preset", choices=list(PRESETS.keys()))
                s_o = gr.Slider(0, 1, 0.5, label="O"); s_c = gr.Slider(0, 1, 0.5, label="C"); s_e = gr.Slider(0, 1, 0.5, label="E"); s_a = gr.Slider(0, 1, 0.5, label="A"); s_n = gr.Slider(0, 1, 0.8, label="N")
                avatar_char = gr.Dropdown(label="Select Character", choices=list(AVATAR_MAP.keys()) if AVATAR_MAP else ["None"], value=default_char)
                show_avatar = gr.Checkbox(label="Show Patient Face", value=True)

            with gr.Group():
                gr.Markdown("### 🪝 Hook Configuration")
                target_layer = gr.Slider(0, 31, 16, step=1, label="Target Injection Layer")
                dual_mode = gr.Checkbox(label="Dual Injection Mode", value=False)
                target_layer_2 = gr.Slider(0, 31, 24, step=1, label="Target Injection Layer 2", visible=False)
                dist_mode = gr.Checkbox(label="Distributed Mode (Multi-Layer)", value=True)
                emo_mode = gr.Checkbox(label="Strict emotion label", value=True)
                intensity = gr.Slider(0.1, 4.0, 1.5, label="Steering Alpha (Intensity)")
                temp = gr.Slider(0.1, 1.5, 0.7, label="Temperature")
                tokens = gr.Slider(64, 1020, 350, label="Max Tokens")
                situation_input = gr.Textbox(label="Problem List", value="Generic clinical case")
                sys_msg = gr.Textbox(label="System Prompt", lines=2)
            
            btn_reset = gr.Button("🔄 Reset Simulation", variant="stop")

        with gr.Column(scale=2):
            dash = gr.HTML(value=build_dashboard(-0.1, 0.0, 0.6, -0.1, 0.0, 0.6, "Gratitude"))
            chatbot = gr.Chatbot(label="Dialogue History", height=500) 
            msg_input = gr.Textbox(label="Doctor Message", placeholder="Type here...", interactive=True, max_length=MAX_CHARS)
            char_counter = gr.Markdown(f"**0**/{MAX_CHARS} characters")

        with gr.Column(scale=1):
            live_plot = gr.Plot(label="Live PAD Trajectory")
            patient_image = gr.Image(value=get_face_image(default_char, "Gratitude"), label="Patient Emotion", interactive=False, visible=True, height=250)
            btn_export = gr.Button("📊 Export Data", variant="primary"); 
            file_download = gr.File(label="Download")
            btn_conclude = gr.Button("🏁 Clinical Debrief", variant="stop")

    # Connect Events
    show_avatar.change(fn=lambda show: gr.update(visible=show), inputs=show_avatar, outputs=patient_image)

    dual_mode.change(fn=lambda show: gr.update(visible=show), inputs=dual_mode, outputs=target_layer_2)

    btn_open_scen.click(fn=lambda: gr.update(visible=True), outputs=scenario_panel)
    scenario_drop.change(fn=lambda n: all_scenarios.get(n, {}).get("DESCRIPTION", ""), inputs=scenario_drop, outputs=scenario_desc)
    btn_cancel_scen.click(fn=lambda: gr.update(visible=False), outputs=scenario_panel)
    
    preset_dd.change(fn=lambda p: PRESETS[p], inputs=[preset_dd], outputs=[s_o, s_c, s_e, s_a, s_n])

    def update_initial_emotion(emo_name, char_name):
        coords = EMOTION_MAP.get(emo_name, [-0.1, 0.0, 0.6])
        new_face = get_face_image(char_name, emo_name)
        # Returns: New PAD State, New Dashboard HTML
        return coords, build_dashboard(*coords, *coords, emo_name), new_face

    # Trigger the update when the user changes the dropdown
    start_emo_dd.change(
        fn=update_initial_emotion, 
        inputs=[start_emo_dd, avatar_char], 
        outputs=[pad_state, dash, patient_image]
    )

    avatar_char.change(
        fn=get_face_image,
        inputs=[avatar_char, start_emo_dd],
        outputs=[patient_image]
    )

    def apply_scenario(name, char_name):
        scen = all_scenarios.get(name, {})
        sys_p = scen.get("SYSTEM PROMPT", "")
        prob_list = scen.get("PROBLEM LIST", "Unknown medical condition.")
        if "### OUTPUT FORMAT" not in sys_p:
            sys_p += "\n\n### OUTPUT FORMAT (STRICT)\n[LINGUISTIC ANALYSIS]: ...\n[INTERNAL THOUGHT]: ...\n[EMOTIONAL STATE]: ...\n[PATIENT RESPONSE]: ..."
        
        emo = scen.get("STARTING EMOTION", "Gratitude")
        coords = EMOTION_MAP.get(emo, [-0.1, 0, 0.6])
        preset = scen.get("PRESET", "Alex (Anxious)")
        p_vals = PRESETS.get(preset, [0.5, 0.5, 0.4, 0.4, 0.8])
        new_face = get_face_image(char_name, emo)
        return gr.update(visible=False), sys_p, prob_list, emo, preset, *p_vals, [], build_dashboard(*coords, *coords, emo), coords, None, []

    btn_confirm_scen.click(fn=apply_scenario, inputs=[scenario_drop, avatar_char], outputs=[scenario_panel, sys_msg, situation_input, start_emo_dd, preset_dd, s_o, s_c, s_e, s_a, s_n, chatbot, dash, pad_state, live_plot, mood_history_state, patient_image])
    
    def update_counter(text):
        count = len(text)
        color = "gray"
        if count > MAX_CHARS * 0.8: color = "orange"
        if count >= MAX_CHARS: color = "red"
        return f"<span style='color:{color}; font-weight:bold;'>{count}</span>/{MAX_CHARS} characters"

    msg_input.change(fn=update_counter, inputs=[msg_input], outputs=[char_counter])

    msg_input.submit(
        patient_respond, 
        [msg_input, chatbot, s_o, s_c, s_e, s_a, s_n, sys_msg, temp, tokens, gr.State(1.15), intensity, pad_state, mood_history_state, situation_input, target_layer, target_layer_2, dual_mode, dist_mode, emo_mode, avatar_char], 
        [chatbot, dash, msg_input, pad_state, live_plot, mood_history_state, file_download, patient_image]
    )


    btn_export.click(fn=export_session_json, inputs=[chatbot, mood_history_state, s_o, s_c, s_e, s_a, s_n,situation_input], outputs=file_download)
    btn_reset.click(fn=lambda: ([], build_dashboard(-0.1,0,0.6, -0.1,0,0.6, "Gratitude"), "", [-0.1,0,0.6], None, []), outputs=[chatbot, dash, msg_input, pad_state, live_plot, mood_history_state, patient_image])
    btn_conclude.click(fn=lambda h, m, o, c, e, a, n: (h + generate_commentary(h, m, "Final"), None), inputs=[chatbot, mood_history_state, s_o, s_c, s_e, s_a, s_n], outputs=[chatbot, file_download])

# demo.launch(server_name="0.0.0.0", server_port=7861)
demo.launch(theme=gr.themes.Soft(),server_port=7861)