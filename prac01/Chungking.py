# 내부 텍스트 문서를 청킹하여 질문에 맞는 내용을 답하는 코드
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다!")

# 파일 읽기
file_path = r"C:\Users\BGR_NC_2_NOTE\Documents\카카오톡 받은 파일\AI_rehabilitation_trainer_data_250411.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} 파일을 찾을 수 없습니다!")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 청크 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_text(text)
print(f"총 {len(texts)}개의 청크 생성됨")

# 벡터스토어 생성
embeddings = HuggingFaceEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
db = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_db")
print("ChromaDB에 청크 저장 완료")

# LLM 및 검색기 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
retriever = db.as_retriever(search_kwargs={"k": 5})

# ChatPromptTemplate으로 프롬프트 정의 (조건 완화)
prompt_template = ChatPromptTemplate.from_template("""
당신은 전문 재활 트레이너 어시스턴트입니다. 다음 내용을 참고하여 질문에 답변해주세요.

[참고 내용]
{context}

[질문]
{question}

참고 내용을 바탕으로 질문에 대해 최대한 도움이 되는 답변을 해주세요.
참고 내용에 정확히 일치하는 정보가 없더라도, 관련된 내용을 바탕으로 유용한 정보를 제공해주세요.
""")

# 간단한 QA 함수
def ask_question(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    messages = prompt_template.format_messages(
        context=context,
        question=query
    )
    
    response = llm.invoke(messages)
    return response.content

# 질의응답 루프
print("질의응답 시작 (종료: exit)")
while True:
    query = input("질문: ")
    if query.lower() == "exit":
        break
    answer = ask_question(query)
    print("답변:", answer)