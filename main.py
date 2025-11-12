import faiss

from sentence_transformers import SentenceTransformer

# GET QUESTIONS AND ANSWERS / IN FUTURE NOT USING TEXT
q = ["123", "222", "2222"]
a = ["22", "12345", "-2222"]


# USING MODEL MiniLM
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ENCODING TEXT / IN FUTURE GETTING VECTORS FROM DB
questions_vectors = model.encode(q, normalize_embeddings=True)
answers_vectors = model.encode(a, normalize_embeddings=True)

# THIS MODEL IS USING 384 SHAPE TENSOR
index = faiss.IndexFlatIP(384)  
index.add(answers_vectors)  

# TOP-K SIMILAR ANSWERS
k = 1 


# SCORES TOP-2 EX: [[0.7, O.3],
#                    [0.12, 0.233]]

# INDICES TOP-2 EX: [[3, 2], 
#                   [2, 1]]

scores, indices = index.search(questions_vectors, k)

# GETTING INDEX OF QUESTION AND INDEX OF ANSWER TO IT (WORKING FOR TOP-K)
for q_index, question in enumerate(questions_vectors):
    print(f"question id: {q_index+1}")
    for rank in range(k):
        idx = indices[q_index][rank]          
        similar = scores[q_index][rank]         
        print(f"{rank+1}) id: {idx+1} - сходство {similar:.4f} ")
    print("-" * 30)


