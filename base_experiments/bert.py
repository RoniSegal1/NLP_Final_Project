import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, pipeline, BertConfig
)
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

print("----- START -----")
MODEL_NAME = "google-bert/bert-base-uncased"
TEST_PATH = "/vol/joberant_nobck/data/NLP_368307701_2425b/maozhaim/datasets/test_dataset.csv"
TRAIN_PATH = "/vol/joberant_nobck/data/NLP_368307701_2425b/maozhaim/datasets/train_dataset.csv"
OUTPUT_DIR = "/home/joberant/NLP_2425b/ronisegal/NLP/bert/fine-tuning-results"
MODEL_SAVE_PATH = "/home/joberant/NLP_2425b/ronisegal/NLP/bert/bert-base-uncased-finetuned-ai-human"
MAX_LEN = 512
EPOCHS = 7
LEARNING_RATE = 2e-5


print("--- GPU check ---")
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print()


print("--- Model and tokenizer configurations ---")
model_name = MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(
    model_name,
    num_labels=2,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.,
    id2label={1: "AI", 0: "Human"},
    label2id={"AI": 1, "Human": 0},
)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)
print()


print("--- Load test dataset ---")
print(f"Dataset path - {TEST_PATH}")
df_test = pd.read_csv(TEST_PATH)
test_dataset = Dataset.from_pandas(df_test)
print(f"Test samples - {len(test_dataset)}")


texts = [example['text'] for example in test_dataset]
true_labels = [int(example['label']) for example in test_dataset]
print()


print("--- Zero-shot testing ---")
pre_pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN
)


print("Calculating zero-shot results")
pre_results = pre_pipe(texts, batch_size=8)
pre_pred_labels = [1 if r['label'] == 'AI' else 0 for r in pre_results]
accuracy = accuracy_score(true_labels, pre_pred_labels)
precision = precision_score(true_labels, pre_pred_labels)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print()




print("--- Load training dataset ---")
print(f"Dataset path - {TRAIN_PATH}")
df = pd.read_csv(TRAIN_PATH)
df_train, df_val = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)
print(f"Train samples - {len(train_dataset)}")
print(f"Validation samples - {len(val_dataset)}")
print()




print("--- Tokenization ---")
def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=MAX_LEN)


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
print()


print("--- Define fine-tuning arguments ---")
fine_tuning_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",      
    learning_rate=LEARNING_RATE,          
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2
)
print()


print("--- Define metrics computations ---")
def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0, pos_label=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
print()




print("--- Define fine-tuning trainer ---")
fine_tuning_trainer = Trainer(
    model=model,
    args=fine_tuning_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)
print()




print("--- Fine-tuning on base model ---")
fine_tuning_trainer.train()
print("Finished")
print()




print("--- Save model ---")
fine_tuning_trainer.save_model(MODEL_SAVE_PATH)
print("Model saved")
print()




print("--- Fine-tuned testing ---")
fine_tuning_pipe = pipeline(
    "text-classification",
    model=MODEL_SAVE_PATH,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN
)


print("Calculating fine-tuning results")
fine_tuning_results = fine_tuning_pipe(texts, batch_size=8)
fine_tuning_pred_labels = [1 if r['label'] == 'AI' else 0 for r in fine_tuning_results]
accuracy = accuracy_score(true_labels, fine_tuning_pred_labels)
precision = precision_score(true_labels, fine_tuning_pred_labels)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print()


print("----- FINISH -----")
