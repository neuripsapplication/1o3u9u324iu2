import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

"""
==================== USER INSTRUCTIONS ====================

This script encodes text data from a CSV file using a SentenceTransformer model.
Before running, make sure to update the file paths if needed:

1. Input file: './ras_texbook_sample.csv' 
2. Output file: './embeddings_sample_jinai_neurips.csv' — will save the resulting embeddings.

You can change the model by modifying the `SentenceTransformer(...)` line.

===========================================================
"""


# Linq-AI-Research/Linq-Embed-Mistral, intfloat/multilingual-e5-large-instruct
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

df = pd.read_csv("./ras_textbook_sample.csv")

df['Chunk_Text_embedding'] = None

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row['Chunk_Text']) if pd.notna(row['Chunk_Text']) else ""
    embeddings = model.encode(text, normalize_embeddings=True)#convert_to_numpy=True
    df.at[i, 'Chunk_Text_embedding'] = embeddings.tolist()

# Save to CSV
df.to_csv("./embeddings_sample_jinai_neurips.csv", index=False)