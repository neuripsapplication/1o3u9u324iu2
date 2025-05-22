import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm.auto import tqdm

# === Config ===
model_id = "meta-llama/Llama-3.3-70B-Instruct"
model_name = model_id.split('/')[-1]

print(f"Loading {model_name}...")
fine_tuned_model_path = f"../../surg_qa_{model_name}"

output_file = f"./gen_q_{model_name}.csv"
num_questions = 1000

if "gemma-3" in model_name:
    attn_imp = "eager"
else:
    attn_imp = None

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_imp
)

model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
model.eval()

# === Define prompt ===
system_prompt = """
You are a surgical expert specializing in robotic-assisted surgery.
Your task is to generate an exam question to evaluate medical students' knowledge of robotic surgery.
Do not generate questions asking about the book title or author names, and when you want to ask questions about steps in a procedure,
use the name of the step instead of step number. For instance, if the chunk has "Step 10. Transection of the distal esophagus",
ask about transection of the distal esophagus instead of step 10.
"""

user_prompt = """
### Question:
Please generate a question for a robotic surgery exam, given this part of the textbook,
Chapter: {}
Chunk: {}
"""

# === Load Textbook Dataset ===
ras_textbook_df = pd.read_csv("../ras_textbook_sample.csv")
rand_chunks = ras_textbook_df[['Chapter', 'Chunk_Text']]

# Store both chunks and their corresponding questions
data = []
for _, chunk in tqdm(rand_chunks.iterrows(), total=len(rand_chunks), desc="Generating questions"):
    if len(chunk['Chunk_Text']) < 100:
        continue
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(chunk['Chapter'], chunk['Chunk_Text'])}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate a question
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("\n")[-1].strip()

    print(f"\n🧠 Chapter: {chunk['Chapter']},\nChunk: {chunk['Chunk_Text']},\nGenerated Question: {response}\n")

    data.append((chunk['Chapter'], chunk['Chunk_Text'], response))

# Create a DataFrame and save as CSV
df_out = pd.DataFrame(data, columns=["Chapter", "Chunk", "Generated_Question"])
df_out.to_csv(output_file, index=False)

print(f"\n✅ Saved {len(df_out)} questions to {output_file}")
