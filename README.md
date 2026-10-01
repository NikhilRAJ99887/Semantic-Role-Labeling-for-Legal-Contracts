# BERT Semantic Role Labeling for Freelance Legal Contracts

## Project goal
Build a BERT-based NER/token-classification system that reads freelance software-development contract clauses and extracts information that can be mapped to three high-level semantic roles:

- WHO: PERSON, PARTY, ROLE
- WHAT: ACTION, DELIVERABLE, PAYMENT, AMOUNT, CONDITION, NOTICE
- WHEN: DATE, DEADLINE, DURATION

The supplied dataset contains 1,200 synthetic contract sentences with BIO annotations: 960 train, 120 validation, 120 test.

> The dataset is synthetic and intended for academic prototyping. Do not describe it as lawyer-verified legal data.

## Folder structure

```text
legal_bert_ner_project/
├── data/
│   ├── legal_contract_ner_large_huggingface.csv
│   ├── legal_contract_ner_large.jsonl
│   └── label_list.txt
├── config.py
├── train.py
├── predict.py
├── run_predict.py
├── app.py
├── requirements.txt
└── README.md
```

## 1. Create environment

Recommended: Python 3.11 or 3.12.

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Then:
```bash
pip install -r requirements.txt
```

## 2. Train BERT

From the project folder:

```bash
python train.py
```

The first run downloads `bert-base-uncased` from Hugging Face. Training creates:

```text
outputs/legal-bert-ner/
```

with the fine-tuned model and tokenizer.

## 3. Test one sentence

```bash
python predict.py
```

Example:

```text
Meera Iyer shall deliver the user interface within three days.
```

Expected semantic categories:

```text
WHO   -> Meera Iyer
WHAT  -> deliver / user interface
WHEN  -> three days
```

Exact predictions depend on the trained model.

## 4. Run the web app

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## 5. Project architecture

```text
Contract PDF/Text
      |
      v
Text extraction / input
      |
      v
Sentence + token preparation
      |
      v
BERT tokenizer
      |
      v
Fine-tuned BERT Token Classifier
      |
      v
BIO entity predictions
      |
      v
Semantic role mapping
      |
      +----------+----------+
      |          |          |
     WHO        WHAT       WHEN
      |          |          |
 PERSON/PARTY ACTION/...  DATE/DEADLINE/DURATION
```

## Important academic note

NER identifies spans and entity types. The WHO/WHAT/WHEN layer in this implementation is a transparent rule-based semantic mapping on top of NER. For a stronger research version, add relation/SRL labels that explicitly connect an actor to an action, object, recipient, and time.
