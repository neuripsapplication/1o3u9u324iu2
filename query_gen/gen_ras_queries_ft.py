import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import login
from datasets import Dataset, DatasetDict
import torch
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

##### Load Model #####
model_id = "meta-llama/Llama-3.3-70B-Instruct"
model_name = model_id.split('/')[-1]

print(f"Loading {model_name}...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

print(f"Loading {model_name} with LoRA...")

# LoRA config
lora_flag = True
if lora_flag:
    peft_config = LoraConfig(
        lora_alpha=64,                           # Scaling factor for LoRA
        lora_dropout=0.05,                       # Add slight dropout for regularization
        r=16,                                    # Rank of the LoRA update matrices
        bias="none",                             # No bias reparameterization
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)

print(model.hf_device_map)

print(f"Finished Loading {model_name}!!!")

##### Tokenizer #####

print("Loading Tokenizer...")

# Load tokenizer
processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


# Define prompt style with a fixed instruction and a dynamic response placeholder
system_prompt = """
You are a surgical expert specializing in robotic-assisted surgery.
Your task is to generate an exam question to evaluate medical students' knowledge of robotic surgery.
Do not generate questions asking about the book title or author names, and when you want to ask questions about steps in a procedure,
use the name of the step instead of step number. For instance, if the chunk have "Step 10. Transection of the distal esophagus",
ask about transection of the distal esophagus instead of step 10. 
"""

user_prompt = """
### Question:
Please generate a question for a robotic surgery exam, given this part of the textbook,
Chapter: {}
Chunk: {}
"""

def formatting_prompts_func(examples):
    chapters = examples["Chapter"]
    questions = examples["Question"]
    answers = examples["Answer"]

    prompts = []
    for chapter, question, answer in zip(chapters, questions, answers):
        if not question.endswith(processor.eos_token):
            question += processor.eos_token
        prompt = (
            f"<|system|>\n{system_prompt}\n"
            f"<|user|>\n{user_prompt.format(chapter, answer)}\n"
            f"<|assistant|>\n{question}"
        )
        prompts.append(prompt)
    return prompts

# Load CSV
df = pd.read_csv("./ras_qa_full.csv")

# Split into train/val
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Wrap in DatasetDict
dataset_dict = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(val_df, preserve_index=False)
})

print("Finished Loading Dataset!!!")

##### SFT (Supervised Fine-Tuning) Trainer #####

print("Preparing SFT Trainer...")

output_path = f"../../surg_qa_{model_name}"

# Training Arguments
training_arguments = SFTConfig(
    output_dir=output_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=10,
    warmup_steps=10,
    logging_steps=0.1,
    logging_strategy="steps",
    save_strategy="no",
    learning_rate=5e-5,
    fp16=False,
    bf16=False,
    report_to="none",
)

# Initialize the Trainer with eval_dataset
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    processing_class=processor,
    train_dataset=dataset_dict['train'],
    peft_config=(peft_config if lora_flag else None),
    formatting_func=formatting_prompts_func,
)

print("Start Training!!!")
trainer.train()
trainer.save_model()

##### Sample Inference #####
sample = dataset_dict['validation'][0]

inf_prompt = (
    f"<|system|>\n{system_prompt}\n"
    f"<|user|>\n{user_prompt.format(sample['Chapter'], sample['Answer'])}\n"
)

inputs = processor.apply_chat_template(
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    use_cache=True,
)
response = processor.batch_decode(outputs, skip_special_tokens=True)
print(response[0])

##### Evaluation #####
# print("Evaluating on validation set...")
# metrics = trainer.evaluate()
# print("Evaluation metrics:", metrics)