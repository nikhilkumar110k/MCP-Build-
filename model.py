from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

peft_config = LoraConfig(task_type="SEQ_2_SEQ_LM", r=16, lora_alpha=32, lora_dropout=0.05)
model = get_peft_model(model, peft_config)

dataset = load_dataset("csv", data_files="./mcp -project/dataset/medicine_dataset.csv")["train"]

def preprocess(batch):
    inputs = [
        f"Name: {name}, Category: {category}, Indication: {indication}, Classification: {classification}"
        for name, category, indication, classification in zip(
            batch["Name"], batch["Category"], batch["Indication"], batch["Classification"]
        )
    ]
    outputs = [
        f"Dosage Form: {dosage}, Strength: {strength}, Manufacturer: {manufacturer}"
        for dosage, strength, manufacturer in zip(
            batch["Dosage Form"], batch["Strength"], batch["Manufacturer"]
        )
    ]
    
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(outputs, padding="max_length", truncation=True, max_length=128).input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./mcp -project/med-model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=2,
    save_strategy="epoch", 
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(), 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    eval_dataset=tokenized
)

trainer.train()
model.save_pretrained("./mcp -project/saved_model/finetuned-medicine")
tokenizer.save_pretrained("./mcp -project/saved_model/finetuned-medicine")
