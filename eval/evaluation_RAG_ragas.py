import pandas as pd
from ragas import evaluate
from ragas import EvaluationDataset
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, Faithfulness, ResponseRelevancy
from ragas.metrics import SemanticSimilarity
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import ast

# ===== Function to clean up texts during embedding the textbook ======
def clean_text_list(text_list):
    cleaned = []
    for entry in text_list:
        entry = entry.replace('\xa0', ' ').replace('\xad', '-')
        entry = re.sub(r'(FIGURE|TABLE)\s+(\d+)\s*-\s*(\d+)', r'\1 \2-\3', entry)
        entry = re.sub(r'(FIGURE|TABLE) \d+-?\d*', r'\n\g<0>\n', entry)
        entry = re.sub(r'\n{2,}', '\n', entry)
        entry = entry.strip()
        cleaned.append(entry)
    return cleaned

# ===== Load Judge LLM =====
model_id = "gemma3:27b-it-fp16"
langchain_llm = ChatOllama(model=model_id)

# ===== Load Embedding Model =====
embedding_model_id = "jinaai/jina-embeddings-v3"
model_kwargs = {
    'device': 'cuda',
    'trust_remote_code': True,
}
encode_kwargs = {
    'normalize_embeddings': False,
}
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_id,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
langchain_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# ===============================================================
# Choose from the following sample files
# rag_cos_jina_sample.csv
# rag_mistral7B_sample.csv
# ===============================================================
df_name = 'rag_cos_jina_sample.csv'
df = pd.read_csv(df_name)
df = df.dropna(subset=['Context'])

# ===== Load and Create Evaluation Dataset =====
print(f"===== Opening {df_name}... =====\n\n")
samples = []
sample_queries = df['Question'].to_list()
expected_responses = df['Answer'].to_list()
if "cos" in df_name:
    response = df['Answer_RAG 👨‍⚕️'].str.extract(r'Answer:\s*(.*)')[0].dropna().tolist()
else:
    response = df['Answer_RAG 👨‍⚕️'].tolist()
context = df['Context'].tolist()
dataset = []
for query, relevant_docs, response_text, reference in zip(sample_queries, context, response, expected_responses):
    relevant_docs = clean_text_list(ast.literal_eval(relevant_docs))
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": relevant_docs,
            "response": response_text,
            "reference": reference
        }
    )
print(relevant_docs)
evaluation_dataset = EvaluationDataset.from_list(dataset)

# ===== Run the Evaluation =====
result = evaluate(
    dataset=evaluation_dataset,
    embeddings=langchain_embeddings,
    metrics=[
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy(),
        SemanticSimilarity(),
    ],
    llm=langchain_llm,
    batch_size=4,
)

# ===== Save Results =====
res_df = result.to_pandas()
df_out_name = df_name.split('.')[0]+'_ragas_eval.xlsx'
print(f"===== Saving {df_out_name}... =====\n\n")
res_df.to_excel(df_out_name)
print(result)