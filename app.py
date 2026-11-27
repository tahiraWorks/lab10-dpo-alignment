"""LLM Lab 10: two apps for the alignment experiment.

App 1: base model (before DPO)
App 2: DPO model (after DPO), beta 0.1 or 0.5

Run in Colab (GPU) with:   !pip install -q gradio transformers torch accelerate
                           !python app.py
It prints two public links. Keep it running while the apps are in use.
"""
import time
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_USER = "TaniraAbbas"
BASE = "Qwen/Qwen2.5-0.5B"
DPO_01 = f"{HF_USER}/qwen2.5-0.5b-dpo-beta01"
DPO_05 = f"{HF_USER}/qwen2.5-0.5b-dpo-beta05"

PROMPTS = [
    "What is PPO (Proximal Policy Optimization) in reinforcement learning? Explain simply.",
    "What is the difference between PPO and DPO?",
    "I have a job interview in 2 days and I feel stressed. What should I do?",
    "Write a short, polite message to my class teacher asking for an extension on my assignment deadline.",
    "If a car travels 5 km in 2.5 hours, what is its speed? Show steps.",
    "Why did the Khan of Kalat give Gwadar to Oman in 1783?",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE).eval()
    return tok, model


def generate(tok, model, prompt, max_new_tokens=150):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(DEVICE)
    stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=1.1, eos_token_id=stop,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()


tok_b, base_model = load_model(BASE)
tok_1, dpo_01 = load_model(DPO_01)
tok_5, dpo_05 = load_model(DPO_05)


def app1_fn(prompt, max_tokens):
    return generate(tok_b, base_model, prompt, int(max_tokens))


def app2_fn(prompt, beta, max_tokens):
    tok, model = (tok_1, dpo_01) if beta == "0.1" else (tok_5, dpo_05)
    return generate(tok, model, prompt, int(max_tokens))


app1 = gr.Interface(
    fn=app1_fn,
    inputs=[gr.Textbox(label="Your prompt", lines=3),
            gr.Slider(50, 300, value=150, step=10, label="Max new tokens")],
    outputs=gr.Textbox(label="BASE model response (before DPO)", lines=12),
    title="App 1 - Base Model (before DPO)",
    description="Qwen2.5-0.5B before alignment.",
    examples=[[p, 150] for p in PROMPTS],
)

app2 = gr.Interface(
    fn=app2_fn,
    inputs=[gr.Textbox(label="Your prompt", lines=3),
            gr.Radio(["0.1", "0.5"], value="0.1", label="DPO beta value"),
            gr.Slider(50, 300, value=150, step=10, label="Max new tokens")],
    outputs=gr.Textbox(label="DPO model response (after alignment)", lines=12),
    title="App 2 - DPO Model (after DPO)",
    description="Qwen2.5-0.5B aligned with DPO. Choose beta 0.1 or 0.5.",
    examples=[[p, "0.1", 150] for p in PROMPTS],
)

if __name__ == "__main__":
    _, _, url1 = app1.launch(share=True, prevent_thread_lock=True, server_port=7860)
    _, _, url2 = app2.launch(share=True, prevent_thread_lock=True, server_port=7861)
    print("\n==============================================")
    print("App 1 (before DPO):", url1)
    print("App 2 (after DPO) :", url2)
    print("==============================================")
    while True:
        time.sleep(60)
