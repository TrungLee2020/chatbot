import os
from docx import Document
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama

class QASystem:
    def __init__(self, file_path, model_path):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.qa_pairs = self.extract_qa_pairs(file_path)
        self.questions = [pair[0] for pair in self.qa_pairs]
        self.question_embeddings = self.model.encode(self.questions)
        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4)

    def extract_qa_pairs(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.docx':
            return self.extract_qa_pairs_from_docx(file_path)
        elif file_extension.lower() == '.pdf':
            return self.extract_qa_pairs_from_pdf(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .docx or .pdf")

    def extract_qa_pairs_from_docx(self, docx_path):
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

    def extract_qa_pairs_from_pdf(self, pdf_path):
        reader = PyPDFLoader(pdf_path)
        qa_pairs = []
        current_question = ""
        for page in reader.pages:
            text = page.extract_text()
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    if line.endswith('?'):
                        current_question = line
                    elif current_question:
                        qa_pairs.append((current_question, line))
                        current_question = ""
        return qa_pairs

    # embedding model
    def get_most_relevant_qa(self, query, top_k=1):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.qa_pairs[i] for i in top_indices]

    # prompt
    def generate_response(self, relevant_qa, question):
        context = "\n".join([f"Q: {qa[0]}\nA: {qa[1]}" for qa in relevant_qa])
        prompt = f"""Dưới đây là thông tin liên quan từ tài liệu:
                {context}
                Dựa vào thông tin trên, hãy trả lời câu hỏi sau một cách chính xác và ngắn gọn:
                Người dùng: {question}
                AI: """
        response = self.llm(prompt, max_tokens=50, stop=["Người dùng:", "\n"], echo=False)
        return response['choices'][0]['text'].strip()


    def answer_question(self, user_input):
        relevant_qa = self.get_most_relevant_qa(user_input)
        query_embedding = self.model.encode([user_input])
        similarity = cosine_similarity(query_embedding, self.model.encode([relevant_qa[0][0]]))[0][0]
        if relevant_qa and similarity > 0.8:
            return relevant_qa[0][1]
        else:
            return self.generate_response(relevant_qa, user_input)


if __name__ == "__main__":
    file_path = 'data/FQA dịch vụ chi hộ.docx'
    model_path = "models/vinallama-7b-chat_q5_0.gguf"
    qa_system = QASystem(file_path, model_path)

    print("Chào bạn! Hãy đặt câu hỏi (gõ 'thoát' để kết thúc).")
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() == 'thoát':
            print("AI: Tạm biệt!")
            break
        response = qa_system.answer_question(user_input)
        print("AI:", response)
