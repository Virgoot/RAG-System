import faiss
import psycopg2
import numpy as np
import ast
import csv
import re
from sentence_transformers import CrossEncoder


# -----------------------------
# 1. Утилиты
# -----------------------------
def parse_vector_str(vstr: str):
    return np.array(ast.literal_eval(vstr), dtype=np.float32)


    # def normalize_spaces(text: str) -> str:
    #     return re.sub(
    #         r"\s+",
    #         " ",
    #         text.replace("\u00A0", " ")
    #             .replace("\u2011", "-")
    #             .replace("\u202F", " ")
    #             .replace("\u2009", " ")
    #             .replace("\u200A", " ")
    #             .replace("\u200B", "")
    #     ).strip()


# -----------------------------
# 2. Загрузка данных из БД
# -----------------------------
conn = psycopg2.connect(
    host="ragdb-rag1.db-msk0.amvera.tech",
    database="RAG",
    user="rag",
    password="rag"
)

with conn.cursor() as cur:
    # вопросы
    cur.execute("SELECT id, q_text, q_vector FROM questions")
    q_rows = cur.fetchall()

    # ответы
    cur.execute("SELECT id, text, c_vector FROM chunks")
    chunk_rows = cur.fetchall()



# Подготовка массивов
question_ids = [r[0] for r in q_rows]
question_texts = [r[1] for r in q_rows]
questions_vectors = np.array([parse_vector_str(r[2]) for r in q_rows], dtype=np.float32)

chunk_ids = [r[0] for r in chunk_rows]
chunk_texts = [r[1] for r in chunk_rows]
chunk_vectors = np.array([parse_vector_str(r[2]) for r in chunk_rows], dtype=np.float32)

# -----------------------------
# 3. FAISS поиск top-k
# -----------------------------
faiss.normalize_L2(questions_vectors)
faiss.normalize_L2(chunk_vectors)

dim = chunk_vectors.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(chunk_vectors)

k = 50
scores, indices = index.search(questions_vectors, k)

# -----------------------------
# 4. CrossEncoder реранк
# -----------------------------
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

final_results = []

batch_size = 16  # можно увеличить для GPU или уменьшить для CPU

for q_idx, retrieved_idxs in enumerate(indices):
    q_text = question_texts[q_idx]

    candidate_chunks = [(chunk_ids[i], chunk_texts[i]) for i in retrieved_idxs]
    pairs = [[q_text, ans_text] for _, ans_text in candidate_chunks]

    # батчинг
    rerank_scores = reranker.predict(pairs, batch_size=batch_size)

    # сортировка по оценкам CrossEncoder
    ranked = sorted(
        zip(candidate_chunks, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    ranked_chunks_ids = [elem[0][0] for elem in ranked]
    final_results.append((question_ids[q_idx], ranked_chunks_ids))



final_answers = []

for q_id, ranked_chunk_ids in final_results:
    # ranked_chunk_ids — список ID
    format_strings = ','.join(['%s'] * len(ranked_chunk_ids[:5]))
    cur.execute(
        f"SELECT answer_id FROM chunks_answers WHERE chunk_id IN ({format_strings})",
        tuple(ranked_chunk_ids[:5])
    )
    answer_ids = cur.fetchall()
    answer_ids_list = [x[0] for x in answer_ids]
    
    final_answers.append((q_id, answer_ids_list))

conn.close()

# -----------------------------
# 5. Сохранение CSV
# -----------------------------
with open("final_result.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["q_id", "ranked_ids"])
    for q_id, ranked_ids_list in final_answers:
        writer.writerow([q_id, answer_ids_list])

