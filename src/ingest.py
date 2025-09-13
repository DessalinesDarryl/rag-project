from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import psycopg2
import textwrap

# Connexion PostgreSQL
conn = psycopg2.connect(
    dbname="ragdb", user="raguser", password="ragpass", host="localhost", port=5432
)
cur = conn.cursor()

# Créer table si elle n'existe pas
cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(384)
)
""")

# dataset RAG
dataset = load_dataset("neural-bridge/rag-dataset-12000", split="train")

# Modèle d'embedding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500):
    return textwrap.wrap(text, chunk_size)

# Ingestion des données
for doc in dataset[:100]:  # limite à 100 
    chunks = chunk_text(doc["text"])
    for chunk in chunks:
        emb = model.encode(chunk).tolist()
        cur.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)", (chunk, emb))

conn.commit()
cur.close()
conn.close()
print("Ingestion terminée")
