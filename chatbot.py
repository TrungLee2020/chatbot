import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Đọc file CSV
df = pd.read_csv('path/to/file/csv')

# Khởi tạo TF-IDF Vectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 2))

# Fit TF-IDF trên các câu hỏi trong dataset
tfidf_matrix = tfidf.fit_transform(df['Question'])

# Khởi tạo mô hình ngôn ngữ cho việc tạo sinh câu trả lời
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForCausalLM.from_pretrained("vinai/phobert-base")

def preprocess_query(query):
    return query.lower()

def find_most_similar_question(query, threshold=0.3):
    processed_query = preprocess_query(query)
    query_vector = tfidf.transform([processed_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_match_index = cosine_similarities.argmax()
    max_similarity = cosine_similarities[best_match_index]

    if max_similarity > threshold:
        return df.iloc[best_match_index]['Question'], df.iloc[best_match_index]['Answer'], max_similarity
    else:
        return max_similarity

def generate_answer(question, context):
    prompt = f"Câu hỏi: {question}\nNgữ cảnh: {context}\n\nCâu trả lời:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_answer.split("Câu trả lời:")[-1].strip()

@app.route('/ask', methods=['POST'])
def ask_question():
    user_input = request.json.get('question')
    similar_question, direct_answer, similarity = find_most_similar_question(user_input)

    if direct_answer:
        generated_answer = generate_answer(user_input, direct_answer)
        response = {
            'similarity': similarity,
            'similar_question': similar_question,
            'direct_answer': direct_answer,
            'generated_answer': generated_answer
        }
    else:
        response = {
            'similarity': similarity,
            'message': 'Không tìm thấy câu hỏi tương tự. Vui lòng thử lại với câu hỏi khác.'
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
