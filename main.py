import re, torch, gradio as gr, pandas as pd, numpy as np
import datetime, os
from config import *
# Import the local models directly
from inference import de_tokenizer, de_model, llama_model, llama_tokenizer, projector
from affect_engine import update_coord, get_mixed_labels
from utils import build_dashboard, log_mood_journey, update_plot, export_session_json, load_all_scenarios

def get_tag(tag, text):
    # This regex now looks for the tag with OR without brackets.
    # It also stops if it sees a new line starting with a CAPITALIZED tag name.
    pattern = rf"\[?{tag}\]?\s*[:\-]?\s*(.*?)(?=\n\s*\[|\n\s*[A-Z\s]+:|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        # Clean up leading punctuation like the annoying comma or extra colons
        content = re.sub(r"^[,\s\:]+", "", content)
        return content
    return ""

def generate_commentary(history, mood_history, situation):
    if not history:
        return [{"role": "assistant", "content": "⚠️ No conversation data available for analysis."}]

    # 1. CLEAN THE TRANSCRIPT (Remove Gradio 5 list/dict junk)
    transcript = ""
    for msg in history:
        role = "Doctor" if msg["role"] == "user" else "Patient"
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
        
        if role == "Patient":
            # Extract only the spoken dialogue for the supervisor
            match = re.search(r"\[PATIENT RESPONSE\]:\s*(.*)", str(content), re.DOTALL)
            content = match.group(1).strip() if match else content
        transcript += f"{role}: {content}\n\n"

    # 2. ENRICHED SUPERVISOR SYSTEM PROMPT
    # Incorporates SPIKES, PAD correlation, and specific evidence requirements
    supervisor_sys = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
ROLE: Expert Medical Educator & Senior Clinical Supervisor.
CONTEXT: A medical student (Doctor) is interacting with a simulated patient (Alex) in a high-stakes scenario.
SCENARIO: {situation}

TASK:
1. Analyze the student's adherence to the SPIKES protocol (Setting, Perception, Invitation, Knowledge, Empathy, Strategy).
2. Correlate the student's words with the P (Pleasure) and A (Arousal) shifts. Final State: {mood_history[-1][4] if mood_history else 'N/A'}.
3. Successes vs. Distress: Identify where the student succeeded and where they caused unnecessary patient distress (evidenced by spikes in Arousal).
4. Evidence: Use specific quotes from the transcript below to justify your feedback.
5. Grading: give the score (0-10) for each aspect for feedback.

Provide a 'CLINICAL FEEDBACK REPORT' that is professional, critical yet constructive.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
TRANSCRIPT:
{transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

[CLINICAL FEEDBACK REPORT]:"""

    # 3. GENERATION WITH TOKEN ISOLATION (Prevents prompt-dumping)
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
    
    # Only decode tokens generated AFTER the supervisor_sys prompt
    raw_feedback = llama_tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True)
    final_report = raw_feedback.split("[CLINICAL FEEDBACK REPORT]:")[-1].strip()
    
    torch.cuda.empty_cache()
    return [{"role": "assistant", "content": f"### 🩺 CLINICAL SUPERVISOR DEBRIEF\n\n{final_report}"}]

def patient_respond(message, history, o, c, e, a, n, sys_p, temp, tokens, penalty, intensity, pad_state, mood_history, situation):
    if not message: return history, gr.update(), "", pad_state, gr.update(), mood_history, None
    prev_p, prev_a, prev_d = pad_state
    
    # Calculate persona baseline
    po, ao, do = (0.21*e + 0.59*a + 0.19*n), (0.15*o + 0.3*a - 0.57*n), (0.25*o + 0.17*c + 0.6*e - 0.32*a)

    with torch.inference_mode():
        # [1] Local Perception
        de_in = de_tokenizer(message, return_tensors="pt").to(DEVICE)
        raw_d = de_model(de_in.input_ids, de_in.attention_mask) 
        dp, da, dd = raw_d[0].tolist()

        # [2] Local Math Engine
        np_val = update_coord(prev_p, dp, po, o, c, e, a, n, "P")
        na_val = update_coord(prev_a, da, ao, o, c, e, a, n, "A")
        nd_val = update_coord(prev_d, dd, do, o, c, e, a, n, "D")
        new_pad = [np_val, na_val, nd_val]
        label = get_mixed_labels(np_val, na_val, nd_val)

        # [3] Build local history prompt
        clean_history_prompt = ""
        for msg in history[-4:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if isinstance(content, list):
                content = " ".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
            
            if role == "assistant":
                # We extract only the spoken response for the prompt context
                match = re.search(r"\[PATIENT RESPONSE\]:\s*(.*)", content, re.DOTALL)
                content = match.group(1).strip() if match else content
            
            clean_history_prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

        fact_guard = f"\n(Context: {situation})"
        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{sys_p}
[CURRENT PAD: P:{np_val:.2f}, A:{na_val:.2f}, D:{nd_val:.2f}]
[EMOTIONAL BLEND: {label}]
{fact_guard}
INSTRUCTION: Follow the pattern: [LINGUISTIC ANALYSIS], [INTERNAL THOUGHT], then [PATIENT RESPONSE].<|eot_id|>
{clean_history_prompt}<|start_header_id|>user<|end_header_id|>

Doctor: {message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

[LINGUISTIC ANALYSIS]:"""

        # [4] Local Generation
        soft_p = projector(torch.tensor([new_pad], device=DEVICE, dtype=torch.bfloat16))
        soft_p = torch.nn.functional.normalize(soft_p, p=2, dim=-1) * intensity
        soft_p = soft_p.unsqueeze(1)
        inputs = llama_tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
        text_embeds = llama_model.get_input_embeddings()(inputs.input_ids).to(torch.bfloat16)
        
        if soft_p.shape[-1] != text_embeds.shape[-1]:
            padding = torch.zeros((1, 1, text_embeds.shape[-1] - soft_p.shape[-1]), device=DEVICE, dtype=torch.bfloat16)
            soft_p = torch.cat([soft_p, padding], dim=-1)

        full_embeds = torch.cat([soft_p, text_embeds, soft_p], dim=1)
        full_attention_mask = torch.cat([torch.ones((1, 1), device=DEVICE), inputs.attention_mask.to(DEVICE), torch.ones((1, 1), device=DEVICE)], dim=1)
    
        out_ids = llama_model.generate(
            inputs_embeds=full_embeds, 
            attention_mask=full_attention_mask,
            max_new_tokens=int(tokens), 
            temperature=float(temp), 
            repetition_penalty=float(penalty), 
            do_sample=True, 
            pad_token_id=llama_tokenizer.pad_token_id
        )

    # [5] Reconstruction - FIX: Decode ONLY the new generated tokens
    # Calculate how many tokens were in the prompt (including the 2 soft_p tokens)

    raw_res = llama_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    
    # Pre-clean: Remove common Llama 3B "stutter" characters at the very start
    raw_res = re.sub(r"^[,\s\:]+", "", raw_res)
    
    # Prepend the tag manually since Llama starts generating content AFTER it
    full_new_text = "[LINGUISTIC ANALYSIS]: " + raw_res
        
    analysis = get_tag('LINGUISTIC ANALYSIS', full_new_text)
    raw_thought = get_tag('INTERNAL THOUGHT', full_new_text)
    ai_label = get_tag('EMOTIONAL STATE', full_new_text)
    response = get_tag('PATIENT RESPONSE', full_new_text)

    # 3B Model Fix: If the model output numbers instead of analysis, 
    # we detect it and provide a generic linguistic reflection.
    if re.search(r"[PAD]\s*[:\-]?\s*-?\d\.", analysis):
        analysis = f"The doctor's mention of '{message[:30]}...' feels clinically cold and heightens my sense of physical vulnerability."

    # FALLBACK: Ensure simulation flow if AI got confused
    if not response or len(response.strip()) < 2:
        response = "I... I can't think. Everything feels so loud. What should i do?"

    display_label = ai_label if (ai_label and len(ai_label) > 2) else label

    final_msg = (
        f"[LINGUISTIC ANALYSIS]: {analysis}\n"
        f"[INTERNAL THOUGHT]: {raw_thought} (Raw ΔP: {dp:.3f}) (Raw ΔA: {da:.3f}) (Raw ΔD: {dd:.3f})\n"
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
    return history, build_dashboard(np_val, na_val, nd_val, prev_p, prev_a, prev_d, display_label), "", new_pad, current_plot, mood_history, csv_path

# --- UI LAYOUT ---
MAX_CHARS = 500
with gr.Blocks() as demo:
    
    all_scenarios = load_all_scenarios()

    with gr.Column(visible=False, variant="panel") as scenario_panel:
        gr.Markdown("## 📂 Select Clinical Scenario")
        with gr.Row():
            scenario_drop = gr.Dropdown(label="Available Scenarios", choices=list(all_scenarios.keys()))
            scenario_desc = gr.Textbox(label="Scenario Description", interactive=False, lines=3)
        
        with gr.Row():
            btn_confirm_scen = gr.Button("✅ Confirm & Reset", variant="primary")
            btn_cancel_scen = gr.Button("❌ Cancel")
    gr.Markdown("# 🩺 Clinical Patient AI Simulator v3.5")

    pad_state = gr.State([-0.1, 0.0, 0.6])
    mood_history_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🛠️ Emotional Setup")
                btn_open_scen = gr.Button("📂 Select Scenario", variant="secondary")
                start_emo_dd = gr.Dropdown(label="Starting Emotion", choices=list(EMOTION_MAP.keys()), value="Gratitude")
                preset_dd = gr.Dropdown(label="Personality Preset", choices=list(PRESETS.keys()))
                s_o = gr.Slider(0, 1, 0.5, label="O"); s_c = gr.Slider(0, 1, 0.5, label="C"); s_e = gr.Slider(0, 1, 0.5, label="E"); s_a = gr.Slider(0, 1, 0.5, label="A"); s_n = gr.Slider(0, 1, 0.8, label="N")
            with gr.Group():
                situation_input = gr.Textbox(label="Problem List", value="Right arm is amputated. Neck is in traction.")
                intensity = gr.Slider(0.5, 5.0, 3.0, label="Intensity"); temp = gr.Slider(0.1, 1.5, 0.7, label="Temp"); tokens = gr.Slider(64, 1060, 350, label="Max Tokens"); penalty = gr.Slider(1.0, 2.0, 1.15, label="Penalty")
                sys_msg = gr.Textbox(label="System Prompt", value="ROLE: Surgical resident named Alex. BIO: Hand injury.", lines=2)
            btn_reset = gr.Button("🔄 Reset Simulation", variant="stop")

        with gr.Column(scale=2):
            dash = gr.HTML(value=build_dashboard(-0.1, 0.0, 0.6, -0.1, 0.0, 0.6, "Gratitude"))
            chatbot = gr.Chatbot(label="Dialogue History", height=500) 
            msg_input = gr.Textbox(label="Doctor Message", placeholder="Type here...", interactive=True, max_length=MAX_CHARS)
            char_counter = gr.Markdown(f"**0**/{MAX_CHARS} characters")

        with gr.Column(scale=1):
            live_plot = gr.Plot(label="Live PAD Trajectory")
            btn_export = gr.Button("📊 Export Report", variant="primary"); 
            file_download = gr.File(label="Download Report")
            btn_conclude = gr.Button("🏁 End & Conclude Test", variant="stop")

    start_emo_dd.change(fn=lambda e: (EMOTION_MAP[e], build_dashboard(*EMOTION_MAP[e], *EMOTION_MAP[e], e)), inputs=[start_emo_dd], outputs=[pad_state, dash])
    preset_dd.change(fn=lambda p: PRESETS[p], inputs=[preset_dd], outputs=[s_o, s_c, s_e, s_a, s_n])
    btn_export.click(fn=export_session_json, inputs=[chatbot, mood_history_state, s_o, s_c, s_e, s_a, s_n, situation_input], outputs=file_download)
    
    def reset_sim(emo):
        torch.cuda.empty_cache()
        coords = EMOTION_MAP[emo]
        return [], build_dashboard(*coords, *coords, emo), "", coords, None, [], None

    # Open Panel
    # btn_open_scen.click(fn=lambda: gr.update(visible=True), outputs=scenario_panel)
    def open_panel_with_desc(current_selection):
        desc = all_scenarios.get(current_selection, {}).get("DESCRIPTION", "Please select a scenario.")
        return gr.update(visible=True), desc

    btn_open_scen.click(
        fn=open_panel_with_desc, 
        inputs=[scenario_drop], 
        outputs=[scenario_panel, scenario_desc]
    )

    def conclude_test_flow(history, mood_history, o, c, e, a, n, situation):
        # 1. Export the files (JSON/CSV) for the student's records
        report_file = export_session_json(history, mood_history, o, c, e, a, n, situation)
        
        # 2. Generate the feedback (this returns a list with the supervisor message)
        feedback_list = generate_commentary(history, mood_history, situation)
        
        # 3. Add a visual separator to the chat for clarity
        history.append({
            "role": "assistant", 
            "content": "--- 🏁 **SIMULATION CONCLUDED - CLINICAL DEBRIEF BELOW** ---"
        })
        
        # 4. Append the actual feedback message to the existing history
        # We take the first (and only) message from the feedback list
        history.append(feedback_list[0])
        
        # 5. FIX: Return 'history' (the full list) instead of 'feedback_history'
        # This keeps the whole conversation visible on the screen
        return history, report_file
        
    btn_conclude.click(
        fn=conclude_test_flow,
        inputs=[chatbot, mood_history_state, s_o, s_c, s_e, s_a, s_n, situation_input],
        outputs=[chatbot, file_download] # Targets the chatbot directly
    )

    # Update description when dropdown changes
    # def update_scen_desc(name):
    #     return all_scenarios.get(name, {}).get("DESCRIPTION", "")
    # scenario_drop.change(fn=update_scen_desc, inputs=scenario_drop, outputs=scenario_desc)
    scenario_drop.change(
        fn=lambda name: all_scenarios.get(name, {}).get("DESCRIPTION", "No description available."), 
        inputs=scenario_drop, 
        outputs=scenario_desc
    )

    # Cancel button
    btn_cancel_scen.click(fn=lambda: gr.update(visible=False), outputs=scenario_panel)

    # Confirm Button Logic (Integrates with your reset_sim logic)
    def apply_scenario(name):
        scen = all_scenarios.get(name)
        if not scen:
            return gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(), 0.5, 0.5, 0.5, 0.5, 0.8, [], gr.update(), [-0.1, 0, 0.6], None, []
        
        # 1. Get the raw System Prompt from the file
        sys_p = scen.get("SYSTEM PROMPT", "")
        
        # 2. SAFETY INJECTION: Ensure the Output Format is present
        # If the parser cut off the format, we add it back here manually.
        if "### OUTPUT FORMAT (STRICT)" not in sys_p:
            sys_p += """
\n### OUTPUT FORMAT (STRICT)
[LINGUISTIC ANALYSIS]: (Identify triggers and explain impact)
[INTERNAL THOUGHT]: (One-sentence psychological reflection)
[EMOTIONAL STATE]: (Label the mood: e.g., Fearful, Stoic, Guarded)
[PATIENT RESPONSE]: (Your spoken dialogue as Patient)"""

        # 3. Emotional Baseline and Preset
        emo = scen.get("STARTING EMOTION", "Denial")
        coords = EMOTION_MAP.get(emo, [-0.1, 0, 0.6])
        preset = scen.get("PRESET", "Alex (Anxious)")
        p_vals = PRESETS.get(preset, [0.5, 0.5, 0.4, 0.4, 0.8])
        
        torch.cuda.empty_cache()
        
        # Return updates (Ensuring sys_msg gets the full sys_p)
        return (
            gr.update(visible=False),              # Hide panel
            sys_p,                                 # FIXED: Now includes Output Format
            scen.get("PROBLEM LIST", ""),          # Problem list
            emo,                                   # Dropdown starting emotion
            preset,                                # Dropdown preset
            p_vals[0], p_vals[1], p_vals[2], p_vals[3], p_vals[4], # O, C, E, A, N
            [],                                    # Reset Dialogue
            build_dashboard(*coords, *coords, emo), # Reset Dashboard
            coords,                                # Reset pad_state
            None,                                  # Clear plot
            []                                     # Clear mood history
        )
    
    def update_counter(text):
        count = len(text)
        color = "gray"
        if count > MAX_CHARS * 0.8: color = "orange"
        if count >= MAX_CHARS: color = "red"
        return f"<span style='color:{color}; font-weight:bold;'>{count}</span>/{MAX_CHARS} characters"

    msg_input.change(fn=update_counter, inputs=[msg_input], outputs=[char_counter])

    # Connect the button to the outputs (Ensure the order matches the 'return' above)
    btn_confirm_scen.click(
        fn=apply_scenario, 
        inputs=scenario_drop, 
        outputs=[
            scenario_panel, sys_msg, situation_input, start_emo_dd, preset_dd,
            s_o, s_c, s_e, s_a, s_n, 
            chatbot, dash, pad_state, live_plot, mood_history_state
        ]
    )
    btn_reset.click(fn=reset_sim, inputs=[start_emo_dd], outputs=[chatbot, dash, msg_input, pad_state, live_plot, mood_history_state, file_download])
    msg_input.submit(patient_respond, [msg_input, chatbot, s_o, s_c, s_e, s_a, s_n, sys_msg, temp, tokens, penalty, intensity, pad_state, mood_history_state, situation_input], [chatbot, dash, msg_input, pad_state, live_plot, mood_history_state, file_download])

# Changed server_name to "127.0.0.1" for standard local use
demo.launch(theme=gr.themes.Soft(),server_name="127.0.0.1", server_port=7860)
# demo.launch(theme=gr.themes.Soft(),server_name="0.0.0.0", server_port=7860)