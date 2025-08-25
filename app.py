import torch, gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "Daregay/ade_pubmedbert_model"  # or "./model" if you uploaded weights into the Space
THR = 0.876  # validation-chosen threshold

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()

@torch.no_grad()
def predict(text, threshold):
    text = (text or "").strip()
    if not text:
        return {"ADE": 0.0, "Not ADE": 1.0}, 0.0  # dict for Label, float for Number

    batch = tok(
        text,
        max_length=320,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    logits = model(**batch).logits
    if logits.shape[-1] == 1:  # binary head with single logit
        p = torch.sigmoid(logits.squeeze()).item()
    else:  # two-logit softmax head
        p = torch.softmax(logits, dim=-1)[0, 1].item()

    probs = {"ADE": float(p), "Not ADE": float(1 - p)}  # what gr.Label expects
    return probs, float(p)

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=5, label="Sentence"),
        gr.Slider(0, 1, value=THR, step=0.001, label="Threshold"),
    ],
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Number(label="P(ADE)"),
    ],
    title="ADE Classifier (PubMedBERT)",
    description="Recall-first classifier; default threshold chosen on validation to reach recall ≥ 0.90.",
    examples=[
        ["The patient developed a rash after starting amoxicillin.", THR],
        ["No adverse reactions were reported during treatment.", THR],
    ],
    cache_examples=False,         # important: avoid startup caching issues
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()