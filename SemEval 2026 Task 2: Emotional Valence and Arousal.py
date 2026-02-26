import transformers

import torch
import torch._dynamo
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datasets import Dataset
import numpy as np
import random
import os
from transformers import set_seed
import torch.nn as nn


def set_all_seeds(seed=42):
    """Set seeds for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_seed(seed)  # HuggingFace transformers seed


def ccc_loss(preds, targets, eps=1e-8):
    preds_mean = torch.mean(preds, dim=0)
    targets_mean = torch.mean(targets, dim=0)

    preds_var = torch.mean((preds - preds_mean) ** 2, dim=0)
    targets_var = torch.mean((targets - targets_mean) ** 2, dim=0)

    cov = torch.mean((preds - preds_mean) * (targets - targets_mean), dim=0)

    ccc = (2 * cov) / (
        preds_var + targets_var + (preds_mean - targets_mean) ** 2 + eps
    )

    return 1 - ccc


class CCCTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        preds = outputs["logits"]

        # Calculate CCC losses for Valence (index 0) and Arousal (index 1)
        ccc_losses = ccc_loss(preds, labels)
        
        # Apply weighting: Arousal is harder, so give it more importance
        # Valence weight: 0.5, Arousal weight: 0.5
        weighted_ccc = (0.5 * ccc_losses[0]) + (0.5 * ccc_losses[1])

        # Lower the MAE weight to 0.01. 
        # Your logs show the model is already in the right numerical range.
        mae = nn.L1Loss()(preds, labels)
        loss = weighted_ccc + 0.01 * mae

        return (loss, outputs) if return_outputs else loss
        

# =========================================================
# The Magic Stopper (Stop at Epoch 4, Plan for 8)
# =========================================================
class EarlyStopAtEpoch(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= 4:
            print(f"\n[CALLBACK] Reached Epoch {state.epoch}. Stopping to lock in 0.56 logic.")
            control.should_training_stop = True
            
set_all_seeds(42)
print("All seeds set to 42 for reproducibility")

# CRITICAL: Disable torch compilation for older GPUs
torch._dynamo.config.suppress_errors = True
torch.backends.cudnn.benchmark = False

# 3. Load the Dataset
df_train = pd.read_csv("https://raw.githubusercontent.com/abdessamadbenlahbib/Datasets/refs/heads/main/train_subtask1.csv")

# 4. Tokenizer and Model Setup
model_name = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model for regression
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    problem_type="regression"
)

# 5. Prepare the Data
df_train = df_train[['user_id', 'text_id', 'text', 'valence', 'arousal']].copy()

# Convert to float32 explicitly
df_train['valence'] = df_train['valence'].astype('float32')
df_train['arousal'] = df_train['arousal'].astype('float32')



# 6. Tokenize and Create Dataset
def preprocess_function(examples):
    # Tokenize
    tokenized = tokenizer(
        examples['text'], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )
    
    # Add labels as float32 numpy arrays
    tokenized['labels'] = np.column_stack([
        np.array(examples['valence'], dtype=np.float32),
        np.array(examples['arousal'], dtype=np.float32)
    ]).tolist()
    
    return tokenized

# Convert to Dataset
train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'valence', 'arousal'])

# Set format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print("Dataset prepared successfully!")

# 7. Define Metrics
def pearson_np(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return (x*y).sum() / (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum()) + 1e-8)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    r_val = pearson_np(labels[:,0], preds[:,0])
    r_aro = pearson_np(labels[:,1], preds[:,1])

    return {
        "r_valence": r_val,
        "r_arousal": r_aro,
        "r_avg": (r_val + r_aro) / 2
    }


# 8. Training Arguments - OPTIMIZED FOR OLDER GPU
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size= 8,  # Reduced for safety
    gradient_accumulation_steps=16,  
    num_train_epochs = 8,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="no",
    save_strategy="no",
    dataloader_num_workers=0,  # Important: avoid multiprocessing issues
    report_to="none",
    #dataloader_pin_memory=False,  # Disable to avoid potential hangs
    seed=42,  # Seed for Trainer
    data_seed=42,  # Seed for data sampling
    # CRITICAL: Disable torch compile for older GPUs
    torch_compile=False,
    use_cpu=False,  # Still use GPU, just without compilation
    gradient_checkpointing=True,
    fp16=True
)

trainer = CCCTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, 
    callbacks=[EarlyStopAtEpoch()] # STOP AT 4
)

# =========================================================
# Train
# =========================================================
print("\nStarting training on full training set...")
trainer.train()
print("Training completed.")

# =========================================================
# Load TEST SET
# =========================================================
df_test = pd.read_csv(
    "https://raw.githubusercontent.com/abdessamadbenlahbib/Datasets/refs/heads/main/test_subtask1.csv"
)

test_dataset = Dataset.from_pandas(df_test.reset_index(drop=True))
test_dataset = test_dataset.map(
    lambda x: tokenizer(
        x["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    ),
    batched=True,
    remove_columns=["text"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"]
)

# =========================================================
# Inference
# =========================================================
print("Running inference on test set...")
predictions = trainer.predict(test_dataset)
preds_1 = predictions.predictions


# ======================================================================================================
# ======================================================================================================


def set_all_seeds(seed=100):
    """Set seeds for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_seed(seed)  # HuggingFace transformers seed


def ccc_loss(preds, targets, eps=1e-8):
    preds_mean = torch.mean(preds, dim=0)
    targets_mean = torch.mean(targets, dim=0)

    preds_var = torch.mean((preds - preds_mean) ** 2, dim=0)
    targets_var = torch.mean((targets - targets_mean) ** 2, dim=0)

    cov = torch.mean((preds - preds_mean) * (targets - targets_mean), dim=0)

    ccc = (2 * cov) / (
        preds_var + targets_var + (preds_mean - targets_mean) ** 2 + eps
    )

    return 1 - ccc


class CCCTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        preds = outputs["logits"]

        # Calculate CCC losses for Valence (index 0) and Arousal (index 1)
        ccc_losses = ccc_loss(preds, labels)
        
        # Apply weighting: Arousal is harder, so give it more importance
        # Valence weight: 0.5, Arousal weight: 0.5
        weighted_ccc = (0.5 * ccc_losses[0]) + (0.5 * ccc_losses[1])

        # Lower the MAE weight to 0.01. 
        # Your logs show the model is already in the right numerical range.
        mae = nn.L1Loss()(preds, labels)
        loss = weighted_ccc + 0.01 * mae

        return (loss, outputs) if return_outputs else loss
        

# =========================================================
# The Magic Stopper (Stop at Epoch 4, Plan for 8)
# =========================================================
class EarlyStopAtEpoch(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= 4:
            print(f"\n[CALLBACK] Reached Epoch {state.epoch}. Stopping to lock in 0.56 logic.")
            control.should_training_stop = True
            
set_all_seeds(100)
print("All seeds set to 100 for reproducibility")

# CRITICAL: Disable torch compilation for older GPUs
torch._dynamo.config.suppress_errors = True
torch.backends.cudnn.benchmark = False

# 3. Load the Dataset
df_train = pd.read_csv("https://raw.githubusercontent.com/abdessamadbenlahbib/Datasets/refs/heads/main/train_subtask1.csv")

# 4. Tokenizer and Model Setup
model_name = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model for regression
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    problem_type="regression"
)

# 5. Prepare the Data
df_train = df_train[['user_id', 'text_id', 'text', 'valence', 'arousal']].copy()

# Convert to float32 explicitly
df_train['valence'] = df_train['valence'].astype('float32')
df_train['arousal'] = df_train['arousal'].astype('float32')



# 6. Tokenize and Create Dataset
def preprocess_function(examples):
    # Tokenize
    tokenized = tokenizer(
        examples['text'], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )
    
    # Add labels as float32 numpy arrays
    tokenized['labels'] = np.column_stack([
        np.array(examples['valence'], dtype=np.float32),
        np.array(examples['arousal'], dtype=np.float32)
    ]).tolist()
    
    return tokenized

# Convert to Dataset
train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'valence', 'arousal'])

# Set format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print("Dataset prepared successfully!")

# 7. Define Metrics
def pearson_np(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return (x*y).sum() / (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum()) + 1e-8)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    r_val = pearson_np(labels[:,0], preds[:,0])
    r_aro = pearson_np(labels[:,1], preds[:,1])

    return {
        "r_valence": r_val,
        "r_arousal": r_aro,
        "r_avg": (r_val + r_aro) / 2
    }


# 8. Training Arguments - OPTIMIZED FOR OLDER GPU
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size= 8,  # Reduced for safety
    gradient_accumulation_steps=16, 
    num_train_epochs = 8,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="no",
    save_strategy="no",
    dataloader_num_workers=0,  # Important: avoid multiprocessing issues
    report_to="none",
    #dataloader_pin_memory=False,  # Disable to avoid potential hangs
    seed=100,  # Seed for Trainer
    data_seed=100,  # Seed for data sampling
    # CRITICAL: Disable torch compile for older GPUs
    torch_compile=False,
    use_cpu=False,  # Still use GPU, just without compilation
    gradient_checkpointing=True,
    fp16=True
)

trainer = CCCTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, 
    callbacks=[EarlyStopAtEpoch()] # STOP AT 4
)

# =========================================================
# Train
# =========================================================
print("\nStarting training on full training set...")
trainer.train()
print("Training completed.")

# =========================================================
# Load TEST SET
# =========================================================
df_test = pd.read_csv(
    "https://raw.githubusercontent.com/abdessamadbenlahbib/Datasets/refs/heads/main/test_subtask1.csv"
)

test_dataset = Dataset.from_pandas(df_test.reset_index(drop=True))
test_dataset = test_dataset.map(
    lambda x: tokenizer(
        x["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    ),
    batched=True,
    remove_columns=["text"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"]
)

# =========================================================
# Inference
# =========================================================
print("Running inference on test set...")
predictions = trainer.predict(test_dataset)
preds_2 = predictions.predictions


# ======================================================================================================
# ======================================================================================================


def set_all_seeds(seed=12345):
    """Set seeds for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_seed(seed)  # HuggingFace transformers seed


def ccc_loss(preds, targets, eps=1e-8):
    preds_mean = torch.mean(preds, dim=0)
    targets_mean = torch.mean(targets, dim=0)

    preds_var = torch.mean((preds - preds_mean) ** 2, dim=0)
    targets_var = torch.mean((targets - targets_mean) ** 2, dim=0)

    cov = torch.mean((preds - preds_mean) * (targets - targets_mean), dim=0)

    ccc = (2 * cov) / (
        preds_var + targets_var + (preds_mean - targets_mean) ** 2 + eps
    )

    return 1 - ccc


class CCCTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        preds = outputs["logits"]

        # Calculate CCC losses for Valence (index 0) and Arousal (index 1)
        ccc_losses = ccc_loss(preds, labels)
        
        # Apply weighting: Arousal is harder, so give it more importance
        # Valence weight: 0.5, Arousal weight: 0.5
        weighted_ccc = (0.5 * ccc_losses[0]) + (0.5 * ccc_losses[1])

        # Lower the MAE weight to 0.01. 
        # Your logs show the model is already in the right numerical range.
        mae = nn.L1Loss()(preds, labels)
        loss = weighted_ccc + 0.01 * mae

        return (loss, outputs) if return_outputs else loss
        

# =========================================================
# The Magic Stopper (Stop at Epoch 4, Plan for 8)
# =========================================================
class EarlyStopAtEpoch(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= 4:
            print(f"\n[CALLBACK] Reached Epoch {state.epoch}. Stopping to lock in 0.56 logic.")
            control.should_training_stop = True
            
set_all_seeds(12345)
print("All seeds set to 12345 for reproducibility")

# CRITICAL: Disable torch compilation for older GPUs
torch._dynamo.config.suppress_errors = True
torch.backends.cudnn.benchmark = False

# 3. Load the Dataset
df_train = pd.read_csv("https://raw.githubusercontent.com/abdessamadbenlahbib/Datasets/refs/heads/main/train_subtask1.csv")

# 4. Tokenizer and Model Setup
model_name = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model for regression
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    problem_type="regression"
)

# 5. Prepare the Data
df_train = df_train[['user_id', 'text_id', 'text', 'valence', 'arousal']].copy()

# Convert to float32 explicitly
df_train['valence'] = df_train['valence'].astype('float32')
df_train['arousal'] = df_train['arousal'].astype('float32')



# 6. Tokenize and Create Dataset
def preprocess_function(examples):
    # Tokenize
    tokenized = tokenizer(
        examples['text'], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )
    
    # Add labels as float32 numpy arrays
    tokenized['labels'] = np.column_stack([
        np.array(examples['valence'], dtype=np.float32),
        np.array(examples['arousal'], dtype=np.float32)
    ]).tolist()
    
    return tokenized

# Convert to Dataset
train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'valence', 'arousal'])

# Set format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print("Dataset prepared successfully!")

# 7. Define Metrics
def pearson_np(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return (x*y).sum() / (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum()) + 1e-8)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    r_val = pearson_np(labels[:,0], preds[:,0])
    r_aro = pearson_np(labels[:,1], preds[:,1])

    return {
        "r_valence": r_val,
        "r_arousal": r_aro,
        "r_avg": (r_val + r_aro) / 2
    }


# 8. Training Arguments - OPTIMIZED FOR OLDER GPU
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size= 8,  # Reduced for safety
    gradient_accumulation_steps=16,  
    num_train_epochs = 8,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="no",
    save_strategy="no",
    dataloader_num_workers=0,  # Important: avoid multiprocessing issues
    report_to="none",
    #dataloader_pin_memory=False,  # Disable to avoid potential hangs
    seed=12345,  # Seed for Trainer
    data_seed=12345,  # Seed for data sampling
    # CRITICAL: Disable torch compile for older GPUs
    torch_compile=False,
    use_cpu=False,  # Still use GPU, just without compilation
    gradient_checkpointing=True,
    fp16=True
)

trainer = CCCTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, 
    callbacks=[EarlyStopAtEpoch()] # STOP AT 4
)

# =========================================================
# Train
# =========================================================
print("\nStarting training on full training set...")
trainer.train()
print("Training completed.")

# =========================================================
# Load TEST SET
# =========================================================
df_test = pd.read_csv(
    "https://raw.githubusercontent.com/abdessamadbenlahbib/Datasets/refs/heads/main/test_subtask1.csv"
)

test_dataset = Dataset.from_pandas(df_test.reset_index(drop=True))
test_dataset = test_dataset.map(
    lambda x: tokenizer(
        x["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    ),
    batched=True,
    remove_columns=["text"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"]
)

# =========================================================
# Inference
# =========================================================
print("Running inference on test set...")
predictions = trainer.predict(test_dataset)
preds_3 = predictions.predictions


# ======================================================================================================
# ======================================================================================================


import zipfile

print("\nFinalizing Ensemble...")
#  Average the raw predictions
# (Assuming your preds are numpy arrays of shape [N, 2])
avg_valence = (preds_1[:, 0] + preds_2[:, 0] + preds_3[:, 0]) / 3
avg_arousal = (preds_1[:, 1] + preds_2[:, 1] + preds_3[:, 1]) / 3

#  Attach to your test dataframe
df_test["pred_valence"] = avg_valence
df_test["pred_arousal"] = avg_arousal

#  Temporal Smoothing (PER USER)
df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
df_test = df_test.sort_values(["user_id", "timestamp"]) # MUST SORT FOR REROLL

df_test["pred_valence"] = df_test.groupby("user_id")["pred_valence"].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
df_test["pred_arousal"] = df_test.groupby("user_id")["pred_arousal"].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

#  Final Clipping
df_test["pred_valence"] = df_test["pred_valence"].clip(-2, 2)
df_test["pred_arousal"] = df_test["pred_arousal"].clip(0, 2)

# =========================================================
#  Final Submission
# =========================================================
df_test = df_test.sort_index() # Back to original test order
output_df = df_test[["user_id", "text_id", "pred_valence", "pred_arousal"]]
output_df.to_csv("pred_subtask1.csv", index=False)

with zipfile.ZipFile("submission_final.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("pred_subtask1.csv")

print("✅ mission accomplished. submission_final.zip is ready for #1.")
