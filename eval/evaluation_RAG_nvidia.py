import pandas as pd
from ragas import evaluate
from ragas import EvaluationDataset
from ragas.metrics import AnswerAccuracy
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

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
# Choose among the following models
# Linq-Embed-Mistral
# GPT-4o
# Mistral-7B-Instruct-v0.3
# ===============================================================
model_name = "Mistral-7B-Instruct-v0.3"

# File containing the QA pairs for Nvidia Answer Accurarcy
df_name = 'ras_qa_nvidia_eval.xlsx'
df = pd.read_excel(df_name, sheet_name=model_name)

print(f"===== Opening {df_name}... =====\n\n")

# ===== Load and Create Evaluation Dataset =====
sample_queries = df['Question'].to_list()
expected_responses = df['Answer_Surgeon'].to_list()
response = df['Answer_Model'].tolist()
dataset = []
for query, reference, response_text in zip(sample_queries, expected_responses, response):
    response_text = " ".join(str(response_text).split()[1:])
    dataset.append(
        {
            "user_input": query,
            "response": response_text,
            "reference": reference,
        }
    )
evaluation_dataset = EvaluationDataset.from_list(dataset)

# ===== Run the Evaluation =====
result = evaluate(
    dataset=evaluation_dataset,
    embeddings=langchain_embeddings,
    metrics=[
        AnswerAccuracy(),
    ],
    llm=langchain_llm,
    batch_size=4,
)

# ===== Save Results =====
res_df = result.to_pandas()
df_out_name = df_name.split('.')[0]+'_'+model_name+'.xlsx'
print(f"===== Saving {df_out_name}... =====\n\n")
res_df.to_excel(df_out_name)
print(result)