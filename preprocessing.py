import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import streamlit as st

login(token=st.secrets.huggingface.token)

# Load langsung dari Hugging Face Model Hub
tokenizer   = AutoTokenizer.from_pretrained("Cheruno/indobert-topic")
model_topic = AutoModelForSequenceClassification.from_pretrained("Cheruno/indobert-topic")
model_sent  = AutoModelForSequenceClassification.from_pretrained("Cheruno/indobert-sentiment")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_topic.to(DEVICE).eval()
model_sent.to(DEVICE).eval()

def _tokenize(txt: str, max_len=128):
    enc = tokenizer.encode_plus(
        txt,  # sudah bersih via normalizer bawaan tokenizer
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)

def predict_topic(text: str) -> str:
    ids, mask = _tokenize(text)
    with torch.no_grad():
        logits = model_topic(ids, attention_mask=mask).logits
    return {0:"Topic 1",1:"Topic 2",2:"Topic 3"}.get(logits.argmax(dim=1).item(), "Unknown")

def predict_sentiment(text: str) -> str:
    ids, mask = _tokenize(text)
    with torch.no_grad():
        logits = model_sent(ids, attention_mask=mask).logits
    return {0:"Negatif",1:"Netral",2:"Positif"}.get(logits.argmax(dim=1).item(), "Unknown")