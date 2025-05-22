"""
==================== USER INSTRUCTIONS ====================

Before running the script, please update the following file paths:
1. `VQA`: Path to your question CSV file (e.g., "YOUR_PATH_HERE/ras_qa_sample.csv")
2. `chpt_path`: Path to your chapter/textbook CSV file (e.g., "YOUR_PATH_HERE/ras_textbook_sample.csv")
3. `embedding_path`: Path to your precomputed phrase embeddings (e.g., "YOUR_PATH_HERE/phrases_embeddings_llama_last.csv")
4. `file_saved`: Name for the output file (will be saved as "YOUR_PATH_HERE/{file_saved}.csv")

Ensure these files exist and are correctly formatted.

===========================================================
"""

import fitz  # PDF handling
from keybert import KeyBERT  # Keyword extraction
import re
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
from difflib import get_close_matches
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from tree_creation_book import TreeRAG
import time

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    print(f'Pytorch version: {torch.__version__}\nCuda available: {torch.cuda.is_available()}\nCuda version: {torch.version.cuda}')
    raise RuntimeError("No GPU found! Ensure you have a compatible GPU and PyTorch installed with CUDA support.")

# ------------------------- Filename normalization -------------------------
def normalize_filename(name):
    name = name.lower()
    name = re.sub(r'[-–—]', '_', name)
    name = re.sub(r'\s+', '', name)
    name = re.sub(r'[^a-z0-9_]', '', name)
    return name

# ------------------------- Match figure name in directory -------------------------
def find_best_match(figure_label, directory):
    figure_label = normalize_filename(figure_label)
    files = os.listdir(directory)
    normalized_files = {normalize_filename(f): f for f in files}
    closest_matches = get_close_matches(figure_label, normalized_files.keys(), n=3, cutoff=0.6)
    return normalized_files.get(closest_matches[0]) if closest_matches else None

# ------------------------- Keyphrase extraction using KeyBERT -------------------------
kw_model = KeyBERT(model="all-mpnet-base-v2")
def extract_keyphrase(text):
    word_count = len(text.split())
    top_n = max(3, word_count // 2)
    print("number kpt: ", top_n)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=top_n)
    if keywords:
        sentence = ' '.join(word for word, _ in keywords)
        return sentence
    return text

# ------------------------- RankLLaMA Chapter-level scoring -------------------------
def CHPT_rank_search(query_keyphrase, list_elements, list_element_kw):
    query_keyphrase = query_keyphrase.lower()
    list_elements = [element.lower() for element in list_elements]
    candidates = []
    for i, element in enumerate(list_elements):
        element_kw = list_element_kw[i]
        inputs = tokenizer_rank(query_keyphrase, element_kw, return_tensors="pt").to(device)
        with torch.no_grad(): 
            outputs = model_rank(**inputs)
        score = outputs.logits[0][0].item()
        candidates.append((score, element))
    return candidates

# ------------------------- RankLLaMA Segment-level scoring -------------------------
def rank_search(query_keyphrase, list_elements):    
    candidates = []
    for element in list_elements:
        inputs = tokenizer_rank(query_keyphrase, element, return_tensors="pt").to(device)
        with torch.no_grad(): 
            outputs = model_rank(**inputs)
        score = outputs.logits[0][0].item()
        candidates.append((score, element))
    return candidates

# ------------------------- Filter by threshold -------------------------
def filter_sentences_by_score(pairs, threshold):
    filtered = [sentence for score, sentence in pairs if score > threshold]
    scores = [score for score, _ in pairs]
    mean_score = max(scores) if scores else 0
    return filtered, mean_score

def filter_pairs_by_score(pairs, threshold):
    return [(score, sentence) for score, sentence in pairs if score > threshold]

# ------------------------- Search for answers in chapter -------------------------
def find_answer_in_chapter(chapter_id, query_keyphrase, tree):
    precise_acc = []
    uncertain_acc = []
    big_text = tree.get_section_by_chapter(chapter_id)
    candidates_btu = rank_search(query_keyphrase, big_text)

    top_btu, _ = filter_sentences_by_score(candidates_btu, 0.7)
    really_precise_top_btu, _ = filter_sentences_by_score(candidates_btu, 6)
    
    if not top_btu:
        return False, precise_acc, uncertain_acc

    for btu_span in top_btu:
        small_units = tree.get_phrases_by_section(chapter_id, btu_span)
        candidates_stu = rank_search(query_keyphrase, small_units)
        
        precise = filter_pairs_by_score(candidates_stu, 4)
        very_precise = filter_pairs_by_score(candidates_stu, 5)
        
        if very_precise:
            print("🍀 Very Precise STU found. 🍀")
            precise_acc.extend(very_precise)
            return True, precise_acc, uncertain_acc
        
        if precise:
            precise_acc.extend(precise)
        
        uncertain = filter_pairs_by_score(candidates_stu, 2.5)
        uncertain_acc.append(uncertain)

    if really_precise_top_btu:
        print("🍀 Super Precise BTU found. 🍀")
        precise_acc.extend(really_precise_top_btu)
        return True, precise_acc, uncertain_acc

    if precise_acc:
        return True, precise_acc, uncertain_acc

    return False, precise_acc, uncertain_acc

# ------------------------- Initialization -------------------------
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
file_saved = "RAG_answers_output"
stubborness = 28
VQA = 'YOUR_PATH_HERE/ras_qa_sample.csv'

# Load ranking model
tokenizer_rank = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', device_map="auto")
base = AutoModelForSequenceClassification.from_pretrained(
    PeftConfig.from_pretrained('castorini/rankllama-v1-7b-lora-passage').base_model_name_or_path,
    num_labels=1, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model_rank = PeftModel.from_pretrained(base, 'castorini/rankllama-v1-7b-lora-passage',
                                       device_map="auto", torch_dtype=torch.float16)
model_rank = model_rank.merge_and_unload().eval()

# Load answer generation model
tokenizer_ans = AutoTokenizer.from_pretrained(model_name)
model_ans = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
print("Answer Generation Model Device Map:", model_ans.hf_device_map)

# Build textbook tree
chpt_path = 'YOUR_PATH_HERE/ras_textbook_sample.csv'
embedding_path = 'YOUR_PATH_HERE/phrases_embeddings_llama_last.csv'
tree = TreeRAG(chpt_path, embedding_path, show_plots=False)
tree_dict = tree.get_forest()
chapters = list(tree_dict.keys())
chapters_kw = [extract_keyphrase(chap) for chap in chapters]

# ------------------------- Main QA Loop -------------------------
df = pd.read_csv(VQA)
questions = df['Question'].dropna().tolist()

start = time.time()
selected = []

for idx, query in enumerate(questions, 1):
    inaccurate_asw = []
    query_keyphrase = extract_keyphrase(query)
    print(f"\n⌛ Question {idx}: {query}\nExtracted Keyphrase: {query_keyphrase}")

    candidates_chpt = CHPT_rank_search(query_keyphrase, chapters, chapters_kw)
    sorted_chpts = sorted(candidates_chpt, key=lambda x: x[0], reverse=True)
    actual_idx = 0
    best_chapter = sorted_chpts[actual_idx][1]
    print(f"🔍 Searching in chapter: {best_chapter}")

    found, precise_hits, uncertain_hits = find_answer_in_chapter(best_chapter, query, tree)
    inaccurate_asw.extend(uncertain_hits)
    uncertain_hits = [item for item in uncertain_hits if item]
    print("unprecise answers", uncertain_hits)

    if found:
        selected = precise_hits
        print("✅ Precise answer found in initial chapter.")
    else:
        found_any = False
        while actual_idx + 1 < len(sorted_chpts) and not found_any and actual_idx < stubborness:
            actual_idx += 1
            chap_id = sorted_chpts[actual_idx][1]
            print(f"🔁 Trying chapter: {chap_id}")
            found, precise_hits, uncertain_hits = find_answer_in_chapter(chap_id, query, tree)
            uncertain_hits = [item for item in uncertain_hits if item]
            print(uncertain_hits)

            if found:
                print("✅ Precise answer found in fallback chapter.")
                selected = precise_hits
                found_any = True
            elif uncertain_hits:
                inaccurate_asw.extend(uncertain_hits)

        if not found_any and len(inaccurate_asw) > 0:
            print("⚠️ No answer found in any chapter — information likely missing.")

    if (len(inaccurate_asw) > 0) and not found:
        print("⚠️ Inaccurate answers found. Showing uncertain hits.")
        selected.extend(inaccurate_asw)
        selected = [item for sub in inaccurate_asw for item in sub]

    final_sorted = sorted(selected, key=lambda x: x[0], reverse=True)
    print("--- Top Answers ---")
    text_list = []

    for item in final_sorted[:5]:
        if isinstance(item, tuple) and len(item) == 2:
            score, text = item
            print(f"[{score:.2f}] {text}")
            text_list.append(text)
        else:
            text = item if isinstance(item, str) else str(item)
            print(f"[??] {text}")
            text_list.append(text)

    merged_text = ' '.join(text_list)

    # Generate final answer using Mistral
    context = merged_text
    pipe = pipeline(
        "text-generation",
        model=model_ans,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        tokenizer=tokenizer_ans,
    )

    messages = [
        {"role": "system", "content": "Answer the following query based on the provided text if useful, answer as if you were the one knowing and you were not reading an answer.\n\n"},
        {"role": "user", "content": f"{context}\n\nQuery: {query}\nAnswer:"},
    ]

    outputs = pipe(messages, max_new_tokens=256, temperature=0.2, repetition_penalty=1.1)
    answer_rag = outputs[0]["generated_text"][-1]["content"]
    print(f"##### RAG ANSWER 🧑‍⚕️:\n{answer_rag}\n")

    # Save results
    df.loc[df['Question'] == query, 'Answer_RAG 👨‍⚕️'] = answer_rag
    matching_indices = df.index[df['Question'] == query]
    if len(matching_indices) == 1:
        df.at[matching_indices[0], 'Context'] = text_list
    else:
        print("Error: query did not match exactly one row")

    df.to_csv(f'YOUR_PATH_HERE/{file_saved}.csv', index=False)
    print(f"Total time: {time.time()-start:.2f}s")
