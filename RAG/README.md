# Hardware Specs

The code was tested on the following device:

One AMD EPYC 7742 64-Core Processor
One Nvidia-A100 (40 GB).

# Installation

Make a copy of the virtual environment:
```
python3 -m venv run_rag
source run_rag/bin/activate
pip install -r requirements.txt
```

# How to run:

- **RankLLaMA+Tree RAG**: open the RAG_NeurIPS.py code, and follow the instructions on the top part of the script, then run the code. This will showcase a demo of our retrival method on the subasample of our dataset that we are sharing (ras_qa_sample, ras_texbook_sample).

- **Cosine Similarity**: open the embedding.py code, and follow the instructions on the top part of the script, then run the code. Do the same for the RAG_cos_NeurIPS.py script.
