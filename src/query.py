import psycopg2
from sentence_transformers import SentenceTransformer
import ollama

# Connexion à Postgres
conn = psycopg2.connect(
    dbname="ragdb", user="raguser", password="ragpass", host="localhost", port=5432
)
cur = conn.cursor()

# Embedding modèle
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

question = "Qu'est-ce que le RAG ?"
q_emb = model.encode(question).tolist()

# Recherche vectorielle
cur.execute("SELECT content FROM documents ORDER BY embedding <-> %s LIMIT 2", (q_emb,))
context = " ".join([row[0] for row in cur.fetchall()])

# Enrichir le prompt
prompt = f"Contexte: {context}\n\nQuestion: {question}\nRéponse:"

# Envoyer à Ollama
response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
print(response["message"]["content"])

cur.close()
conn.close()
