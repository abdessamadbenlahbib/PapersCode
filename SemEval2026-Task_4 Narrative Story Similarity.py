# ==================================================================== TASK A ====================================================================
# ================================================================================================================================================



import json
import torch
import requests
import random
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import gc
import os

# ================= CONFIG =================
GITHUB_RAW_BASE_URL = "https://raw.githubusercontent.com/abdessamadbenlahbib/Datasets/main/"
TRAIN_FILE = "synthetic_data_for_classification.jsonl"
DEV_FILE   = "dev_track_a.jsonl"
TEST_FILE  = "test_track_a.jsonl"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./mnrl_sentence_transformer"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================= UTIL =================
def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def load_jsonl(url, filename):
    r = requests.get(url + filename)
    r.raise_for_status()
    return [json.loads(line) for line in r.text.splitlines()]

# ================= LOAD DATA =================
print("📂 Loading data...")
train_raw = load_jsonl(GITHUB_RAW_BASE_URL, TRAIN_FILE)
dev_raw   = load_jsonl(GITHUB_RAW_BASE_URL, DEV_FILE)
test_raw  = load_jsonl(GITHUB_RAW_BASE_URL, TEST_FILE)

# ================= FILTER DATA =================
def filter_data(data, is_test=False):
    clean = []
    for d in data:
        if all(d.get(k) and str(d[k]).strip() for k in ["anchor_text", "text_a", "text_b"]):
            if is_test or "text_a_is_closer" in d:
                clean.append(d)
    return clean

train_data = filter_data(train_raw)
dev_data   = filter_data(dev_raw)
test_data  = filter_data(test_raw, is_test=True)

print(f"✓ Train samples: {len(train_data)}")
print(f"✓ Dev samples:   {len(dev_data)}")
print(f"✓ Test samples:  {len(test_data)}")

# ================= MODEL =================
model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE,
    trust_remote_code=True
)
model.max_seq_length = MAX_LENGTH

# ================= BUILD MNLR TRIPLETS =================
train_examples = []

for d in train_data:
    anchor = d["anchor_text"]

    if d["text_a_is_closer"]:
        positive = d["text_a"]
        negative = d["text_b"]
    else:
        positive = d["text_b"]
        negative = d["text_a"]

    train_examples.append(
        InputExample(texts=[anchor, positive, negative])
    )

train_loader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=BATCH_SIZE,
    drop_last=True,
    collate_fn=model.smart_batching_collate
)

train_loss = losses.MultipleNegativesRankingLoss(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ================= TRAIN =================
best_f1 = 0.0
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n🟢 Training started...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        sentence_features, labels = batch
        sentence_features = [
            {k: v.to(DEVICE) for k, v in feature.items()}
            for feature in sentence_features
        ]
        labels = labels.to(DEVICE)
        
        loss = train_loss(sentence_features, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # ================= DEV EVAL =================
    model.eval()
    y_true, y_scores = [], []

    with torch.no_grad():
        for d in dev_data:
            anchor = "query: " + d["anchor_text"]

            emb_anchor = model.encode(anchor, normalize_embeddings=True)

            emb_a = model.encode("passage: " + d["text_a"], normalize_embeddings=True)
            emb_b = model.encode("passage: " + d["text_b"], normalize_embeddings=True)

            score = float(np.dot(emb_anchor, emb_a) > np.dot(emb_anchor, emb_b))
            y_scores.append(score)
            y_true.append(1 if d["text_a_is_closer"] else 0)

    acc = accuracy_score(y_true, y_scores)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_scores, average="binary")

    print(f"\nEpoch {epoch+1}")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Dev → Acc: {acc:.4f} | F1: {f1:.4f} | P: {p:.4f} | R: {r:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        model.save(OUTPUT_DIR)
        print("✓ Best model saved")

    clear_gpu()

print("\n🏆 Best Dev F1:", best_f1)

# ================= TEST PREDICTION =================
print("\n📊 Test prediction...")

model = SentenceTransformer(OUTPUT_DIR, device=DEVICE)

predictions = []
with torch.no_grad():
    for d in test_data:
        anchor = "query: " + d["anchor_text"]

        emb_anchor = model.encode(anchor, normalize_embeddings=True)
        emb_a = model.encode("passage: " + d["text_a"], normalize_embeddings=True)
        emb_b = model.encode("passage: " + d["text_b"], normalize_embeddings=True)

        pred = np.dot(emb_anchor, emb_a) > np.dot(emb_anchor, emb_b)
        d["text_a_is_closer"] = bool(pred)

with open("test_predictions_bge.jsonl", "w") as f:
    for d in test_data:
        f.write(json.dumps(d) + "\n")

print("✓ Test predictions saved")


# ==================================================================== TASK B ====================================================================
# ================================================================================================================================================

import json
import requests
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= CONFIG =================
TEST_URL = "https://raw.githubusercontent.com/abdessamadbenlahbib/Datasets/refs/heads/main/test_track_b.jsonl"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
OUTPUT_FILE = "task_b_embeddings.jsonl"

BATCH_SIZE = 16
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD DATA =================
print("📂 Loading test data...")
response = requests.get(TEST_URL)
response.raise_for_status()
test_data = [json.loads(line) for line in response.text.splitlines()]

texts = [d["text"] for d in test_data if d.get("text") and d["text"].strip()]
print(f"✓ Loaded {len(texts)} texts")

# ================= MODEL =================
print("🧠 Loading model...")
model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE,
    trust_remote_code=True
)
model.max_seq_length = MAX_LENGTH
model.eval()

# ================= EMBEDDING =================
print("🟢 Generating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True
)

# ================= SAVE =================
print("💾 Saving embeddings...")
with open(OUTPUT_FILE, "w") as f:
    for text, emb in zip(texts, embeddings):
        record = {
            "text": text,
            "embedding": emb.tolist()
        }
        f.write(json.dumps(record) + "\n")

print("✅ Done.")
print(f"📄 Output file: {OUTPUT_FILE}")

