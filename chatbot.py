from flask import Flask, request, jsonify, send_from_directory
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
from sklearn.feature_extraction.text import TfidfVectorizer
from logs.log import get_logger
import random

# logger
logger = get_logger(__name__)

app = Flask(__name__)

# Tên của bot
BOT_NAME = "Bot Biết Tuốt"

# Các câu trả lời khi không tìm thấy thông tin phù hợp
NO_ANSWER_RESPONSES = [
    "Xin lỗi, tôi không có thông tin về câu hỏi này. Bạn có thể hỏi điều khác không?",
    "Tôi chưa được đào tạo về vấn đề này. Bạn có thể đặt câu hỏi khác không?",
    "Thật tiếc, tôi không thể trả lời câu hỏi này. Hãy thử hỏi điều gì đó khác nhé!"
]
def extract_qa_pairs_from_docx(docx_path):
    doc = Document(docx_path)
    qa_pairs = []
    current_question = ""
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            if text.endswith('?'):
                current_question = text
            elif current_question:
                qa_pairs.append((current_question, text))
                current_question = ""
    return qa_pairs


# Đọc dữ liệu từ file DOCX
docx_path = 'data/FQA dịch vụ chi hộ.docx'
qa_pairs = extract_qa_pairs_from_docx(docx_path)
questions = [pair[0] for pair in qa_pairs]

# Sử dụng TF-IDF để vector hóa DB
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.85, min_df=2)
tfidf_matrix = vectorizer.fit_transform(questions)

# Tải mô hình vinallama cho Q&A
llm = Llama(model_path="models/vinallama-7b-chat_q5_0.gguf", n_ctx=2048, n_threads=4)


def get_most_relevant_qa(query, top_k=1):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [qa_pairs[i] for i in top_indices], similarities[top_indices[0]]


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        user_input = data.get('question')
        logger.info("Câu hỏi nhận được *{}*".format(user_input))

        if not user_input:
            return jsonify({"error": "No question provided"}), 400

        # Xử lý câu hỏi về nội dung
        relevant_qa, similarity = get_most_relevant_qa(user_input)

        if similarity > 0.5:  # If similarity is greater than 50%
            response = relevant_qa[0][1]
        else:
            response = random.choice(NO_ANSWER_RESPONSES)

        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(debug=True)
