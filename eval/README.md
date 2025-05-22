
# Hardware Specs

The code was tested on the following device:

- One AMD EPYC 7742 64-Core Processor  
- Two Nvidia A100 (40 GB)

# Installation

Make a copy of the virtual environment:

```bash
python3 -m venv eval_rag
source eval_rag/bin/activate
pip install -r requirements.txt
```

To run the evaluation codes, please make sure you have installed [Ollama](https://github.com/ollama/ollama?tab=readme-ov-file) and check if it is running in the terminal.

---

## `evaluation_RAG_ragas.py`

Uses a Judge LLM ([Gemma3-27B, Ollama](https://ollama.com/library/gemma3:27b-it-fp16)) and an embedding model ([jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)) to evaluate the metrics described in [RAGAS for RAG evaluation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/).

There are two CSV files:

- `rag_cos_jina_sample.csv`: Classical retrieval with an embedding model and answer generation using [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
- `rag_mistral7B_sample.csv`: Retrieval with our RankLLaMA+Tree-RAG method and answer generation using [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3).

Both methods are used to answer questions related to the three procedures of the textbook for validation. Sample evaluations are provided in this repository (e.g., `rag_cos_jina_sample_ragas_eval.xlsx`, `rag_mistral7B_sample_ragas_eval.xlsx`). The average metrics are shown below, with a variance (1σ) of around ±0.05.

| Model                    | Context Precision | Context Recall | Faithfullness | Answer Relevancy | Semantic Similarity |
| ------------------------ | ----------------- | -------------- | ------------- | ---------------- | ------------------- |
| Mistral-7B-Instruct-v0.3 | 0.9796            | 0.9566         | 0.8517        | 0.9340           | 0.7363              |
| jina-embeddings-v3       | 0.4302            | 0.6051         | 0.6539        | 0.7565           | 0.7223              |

---

## `evaluation_RAG_nvidia.py`

The models used here are the same as above. The file used for this evaluation is `ras_qa_nvidia_eval.xlsx`. It contains three sheets, each with answers from a different model:

- GPT-4o
- Linq-Embed-Mistral + Llama-3.2-1B
- RankLLaMA+Tree-RAG+Mistral-7B

In the code, you can select the model answers to evaluate, generating files like `rag_qa_nvidia_eval_GPT-4o.xlsx`.

We ran the code five times to check metric variance. Results:

| Model                    | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 | Mean    | Std. dev. |
| ------------------------ | ------- | ------- | ------- | ------- | ------- | ------- | --------- |
| Mistral-7B-Instruct-v0.3 | 0.8133  | 0.8127  | 0.8144  | 0.8144  | 0.8127  | 0.8135  | 0.0009    |
| GPT-4                    | 0.4942  | 0.4983  | 0.4975  | 0.4983  | 0.4975  | 0.4972  | 0.0017    |
| Linq-Embed-Mistral       | 0.6634  | 0.6337  | 0.6328  | 0.6328  | 0.6320  | 0.6389  | 0.0137    |
