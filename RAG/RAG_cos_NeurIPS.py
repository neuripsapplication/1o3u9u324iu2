# ============================================
# INSTRUCTIONS
# ============================================
# This script should be run after running `embedding.py`,
# which generates the file: `embeddings_sample_jinai_neurips.csv`.
#
# Update the folder paths below if needed, but keep the filenames:
# 1. "./your_path/embeddings_sample_jinai_neurips.csv" – contains text chunk embeddings
# 2. "./your_path/ras_qa_sample.csv" – contains questions to be answered
# 3. "./your_path/RAG_cos_NeurIPS.csv" – output CSV with generated answers and context
# ============================================

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from ast import literal_eval
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Load the CSV with precomputed embeddings ===
df_embed = pd.read_csv("./your_path/embeddings_sample_jinai_neurips.csv")
for col in ['Chunk_Text_embedding']:
    df_embed[col] = df_embed[col].apply(literal_eval).apply(np.array)

# === Load embedding model ===
embed_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

# === Load answer generation model ===
tokenizer_ans = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model_ans = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device_map="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Answer Generation Model Device Map:", model_ans.hf_device_map)

# === Retrieval Function ===
def retrieve_top_k(query, df, embed_col='Chunk_Text_embedding', k=6):
    query_embedding = embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    doc_embeddings = np.vstack(df[embed_col].values)
    scores = np.dot(doc_embeddings, query_embedding)
    top_k_idx = np.argsort(scores)[::-1][:k]
    return df.iloc[top_k_idx].assign(similarity=scores[top_k_idx])

# === Load questions ===
vqa_path = './your_path/ras_qa_sample.csv'
df_questions = pd.read_csv(vqa_path)
questions = df_questions['Question'].dropna().unique().tolist()
print(f"Loaded {len(questions)} unique questions.")

start_all = time.time()

for idx, query in enumerate(questions, 1):
    print(f"\n=== [{idx}/{len(questions)}] QUERY: {query} ===")

    # Retrieve top-k relevant chunks
    results = retrieve_top_k(query, df_embed, embed_col='Chunk_Text_embedding', k=5)
    context = "\n\n".join(results['Chunk_Text'].tolist())
    text_list = results['Chunk_Text'].tolist()

    # === RAG PROMPT ===
    prompt = (
        f"Answer the following query based on the provided text. Do not explain, just answer:\n\n"
        f"{context}\n\nQuery: {query}\nAnswer:"
    )
    inputs = tokenizer_ans(prompt, return_tensors="pt").to(device)
    out_ids = model_ans.generate(
        **inputs,
        max_new_tokens=75,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer_ans.eos_token_id
    )
    answer_rag = tokenizer_ans.decode(out_ids[0], skip_special_tokens=True)
    print(f"##### RAG ANSWER 👨‍⚕️:\n{answer_rag}\n")

    # === No-RAG Prompt ===
    prompt_nr = f"Answer the following query:\n\nQuery: {query}\nAnswer:"
    inputs_nr = tokenizer_ans(prompt_nr, return_tensors="pt").to(device)
    out2 = model_ans.generate(
        **inputs_nr,
        max_new_tokens=75,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer_ans.eos_token_id
    )
    answer_nr = tokenizer_ans.decode(out2[0], skip_special_tokens=True)
    print(f"##### CLASSIC ANSWER 🗿:\n{answer_nr}\n")

    # === Store results ===
    mask = df_questions['Question'] == query
    if mask.sum() == 1:
        df_questions.loc[mask, 'Answer_RAG 👨‍⚕️'] = answer_rag
        df_questions.loc[mask, 'Answer_no_RAG 🗿'] = answer_nr
        df_questions.loc[mask, 'Context'] = str(text_list)
    else:
        print("❌ Warning: Multiple or no matches found for this question.")

    # Save intermediate results
    df_questions.to_csv('./your_path/RAG_cos_NeurIPS.csv', index=False)

print(f"\n🏁 DONE in {time.time() - start_all:.2f}s")
