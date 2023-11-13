import os 
path = '/Users/shin/Downloads/valiant-imagery-399603-80f2300bb884.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

import os
import sys
import vertexai
import pg8000
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy
from transformers import AutoModel, AutoTokenizer
from vertexai.preview.language_models import ChatModel, InputOutputTextPair, ChatMessage
# TextGenerationModel, InputOutputTextPair, TextEmbeddingModel

current_directory = os.getcwd()
path = current_directory + '/esoteric-stream-399606-6993766aaeea.json'
print(path)
try :
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
    print("Api connected")
except :
    print("Api connect fail : Could not found api key file.")
    sys.exit()

class test :
    def __init__(self):
        self.history = []
        PROJECT_ID = "esoteric-stream-399606"
        LOCATION = "us-central1"

        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # word embedding을 위한 함수
        def get_KoSimCSE():
            model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta-multitask')
            tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')

            return model, tokenizer

        self.model, self.tokenizer = get_KoSimCSE()

        # VectorDB 연결
        instance_connection_name = "esoteric-stream-399606:asia-northeast3:wjdfoek3"
        db_user = "postgres"
        db_pass = "pgvectorwjdfo"
        db_name = "pgvector"

        # initialize Cloud SQL Python Connector object
        connector = Connector()

        def getconn() -> pg8000.dbapi.Connection:
            conn: pg8000.dbapi.Connection = connector.connect(
                instance_connection_name,
                "pg8000",
                user=db_user,
                password=db_pass,
                db=db_name,
                ip_type=IPTypes.PUBLIC,
            )
            return conn

        self.pool = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
        )
        return

    def service(self, query_text):
        chat_model = ChatModel.from_pretrained("chat-bison@001")  #chat model 불러오기

        inputs = self.tokenizer(query_text, padding=True, truncation=True, return_tensors="pt")

        embeddings, _ = self.model(**inputs, return_dict=False)
        embedding_arr = embeddings[0][0].detach().numpy()
        embedding_str = ",".join(str(x) for x in embedding_arr)
        embedding_str = "["+embedding_str+"]"

        insert_stat, param = sqlalchemy.text(
                    """SELECT info, related, insist, content, close FROM minsa
                    ORDER BY v <=> :query_vec LIMIT 1"""
        ), {"query_vec": embedding_str}

        with self.pool.connect() as db_conn: # 쿼리 실행문
            result = db_conn.execute(
                insert_stat, parameters = param
            ).fetchall()

        print(result)
        #query 결과를 문자열로 바꾸기 <- Context에는 문자열만 들어갈 수 있음
        judge = ""
        info = ["판례명", "관련 법령", "주장", "내용", "판결문"]
        for i in range(len(result)) :
            for j in range(len(result[i])) :
                judge += info[j] + result[i][j]
                judge += " "

        print(judge)

        chat_model = ChatModel.from_pretrained("chat-bison@001")

        output_chat = chat_model.start_chat(
            context="법률 자문을 구하는 사용자들에게 기존의 판례들에 기반해 답변해주는 서비스야 주어진 판례를 보고 요약해서 사용자의 상황에 어떻게 적용해야할 지 답변해주거나 사용자의 질문에 맞는 답변을 해줘 판례 : " +judge,
            message_history = self.history,
            temperature=0.3,
            max_output_tokens=1024,
            top_p=0.8,
            top_k=10
        )

        output = output_chat.send_message(query_text).text
        self.history.append(ChatMessage(content = query_text, author = "user"))
        self.history.append(ChatMessage(content = output, author = "bot"))

        return output


from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

question_history = []

@app.route('/', methods=['GET', 'POST'])
def start():
    if request.method == 'GET':
        return jsonify({"question_history": question_history})

    if request.method == 'POST':
        data = request.json
        question = data.get("question")
        question_history.append(question)

        chatbot = test()
        answer = chatbot.service(question)

        
        response_data = {"question": question, "answer": answer}
        return jsonify(response_data)

if __name__ == '__main__':
    app.run(port = 5000)
