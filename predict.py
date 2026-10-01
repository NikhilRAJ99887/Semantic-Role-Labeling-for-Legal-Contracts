import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from config import OUTPUT_DIR


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForTokenClassification.from_pretrained(OUTPUT_DIR)
    model.eval()
    return tokenizer, model


def bio_to_entities(text, tokenizer, model):
    # Fast tokenizer lets us recover character offsets for each word/subword.
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        logits = model(**enc).logits[0]
    pred_ids = torch.argmax(logits, dim=-1).tolist()

    entities = []
    current = None
    for (start, end), pid in zip(offsets, pred_ids):
        if start == end:
            continue
        label = model.config.id2label[int(pid)]
        if label == "O":
            if current:
                entities.append(current)
                current = None
            continue
        prefix, typ = label.split("-", 1)
        if prefix == "B" or current is None or current["type"] != typ:
            if current:
                entities.append(current)
            current = {"type": typ, "start": start, "end": end}
        else:
            current["end"] = end
    if current:
        entities.append(current)

    for e in entities:
        e["text"] = text[e["start"]:e["end"]]
    return entities


def semantic_roles(entities):
    who_types = {"PERSON", "PARTY", "ROLE"}
    what_types = {"ACTION", "DELIVERABLE", "PAYMENT", "AMOUNT", "CONDITION", "NOTICE"}
    when_types = {"DATE", "DEADLINE", "DURATION"}
    result = {"WHO": [], "WHAT": [], "WHEN": []}
    for e in entities:
        if e["type"] in who_types:
            result["WHO"].append(e)
        elif e["type"] in what_types:
            result["WHAT"].append(e)
        elif e["type"] in when_types:
            result["WHEN"].append(e)
    return result


def analyze(text):
    tokenizer, model = load_model()
    entities = bio_to_entities(text, tokenizer, model)
    roles = semantic_roles(entities)
    return entities, roles


if __name__ == "__main__":
    text = input("Enter a contract sentence:\n> ").strip()
    entities, roles = analyze(text)
    print("\nEntities:")
    for e in entities:
        print(f"{e['type']:12} -> {e['text']}")
    print("\nWHO:")
    for e in roles["WHO"]: print("-", e["text"], f"({e['type']})")
    print("\nWHAT:")
    for e in roles["WHAT"]: print("-", e["text"], f"({e['type']})")
    print("\nWHEN:")
    for e in roles["WHEN"]: print("-", e["text"], f"({e['type']})")
