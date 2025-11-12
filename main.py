import faiss
import psycopg2
import numpy as np
import ast
import os

def parse_vector_str(vstr: str):
    """Преобразует строку '[0.1, 0.2, 0.3]' в np.array(float32)."""
    vlist = ast.literal_eval(vstr)  # безопаснее, чем eval()
    return np.array(vlist, dtype=np.float32)

QUESTIONS_FILE = "questions.npy"
ANSWERS_FILE = "answers.npy"

if os.path.exists(QUESTIONS_FILE) and os.path.exists(ANSWERS_FILE):
    questions_vectors = np.load(QUESTIONS_FILE)
    answers_vectors = np.load(ANSWERS_FILE)
else:
    conn = psycopg2.connect(
        host="ragdb-rag1.db-msk0.amvera.tech",
        database="RAG",
        user="rag",
        password="rag"
    )

    with conn.cursor() as cur:
        cur.execute("SELECT q_vector FROM questions")
        questions_vectors = np.array(
            [parse_vector_str(row[0]) for row in cur.fetchall()],
            dtype=np.float32
        )

        cur.execute("SELECT a_vector FROM answers")
        answers_vectors = np.array(
            [parse_vector_str(row[0]) for row in cur.fetchall()],
            dtype=np.float32
        )

    conn.close()

    np.save("questions.npy", questions_vectors)
    np.save("answers.npy", answers_vectors)



# THIS MODEL IS USING 384 SHAPE TENSOR
index = faiss.IndexFlatL2(384)  
index.add(answers_vectors)  

# TOP-K SIMILAR ANSWERS
k = 5


scores, indices = index.search(questions_vectors, k)


# for q_index, question in enumerate(questions_vectors):
#     print(f"question id: {q_index+1}")
#     for rank in range(k):
#         idx = indices[q_index][rank]          
#         similar = scores[q_index][rank]         
#         #print(f"{rank+1}) answer_id: {idx+1} - сходство {similar:.4f} ")
#         data.append((q_index, int(idx)))
#     print("-" * 30)


data = [(q_idx + 1, [int(idx)+1 for idx in indices_row]) 
        for q_idx, indices_row in enumerate(indices) 
        ]


print(data)
