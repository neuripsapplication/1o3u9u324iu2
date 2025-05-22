
# Hardware Specs

The code was tested on the following device:

- 1x AMD EPYC 7742 64-Core Processor  
- 8x Nvidia A100 (40 GB)

# Installation

Make a copy of the virtual environment:

```bash
python3 -m venv query_gen
source query_gen/bin/activate
pip install -r requirements.txt
```

---

## `gen_ras_queries_ft.py`

This code fine-tunes [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) to generate queries related to robotic-assisted surgeries. The full Query-Answer (QA) set curated from the experts are included in `ras_qa_full.csv`.

## `gen_ras_queries_inf.py`

This code uses the fine-tuned [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) to generate queries related to robotic-assisted surgeries. Currently, it is set to use the sample chunks of the textbook included in `ras_textbook_sample.csv`. The generated questions will be saved in a separate file (`gen_q_Llama-3.3-70B-Instruct.csv`).
