import os
from dotenv import load_dotenv
import wikipedia

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage

# 2. 사용자 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(".env에 OPENAI_API_KEY가 없습니다!")

# LLM 설정
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=api_key
)

# Embedding 모델
embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# VectorStore 연결 (자동 persistence)
vectorstore = Chroma(persist_directory="./wiki_chroma_db", embedding_function=embedding_model)


# 3. Wikipedia 검색 함수
def wiki_search(query, num_results=3):
    """Wikipedia에서 검색 후 요약 반환"""
    wikipedia.set_lang("ko")  # 한국어 설정
    results = []
    try:
        search_titles = wikipedia.search(query, results=num_results)
        for title in search_titles:
            page = wikipedia.page(title)
            results.append(page.content[:2000])  # 문서 길이 제한
    except Exception as e:
        print("Wikipedia 검색 실패:", e)
    return results


# 4. 검색 결과 VectorStore 저장
def store_wiki_results(results):
    for i, text in enumerate(results):
        vectorstore.add_texts([text], metadatas=[{"source": f"wiki_{i}"}])


# 5. 질문 처리 (RAG)
def ask_question(question, k=3, save_to_db=True):
    # 1) Wikipedia 검색
    wiki_results = wiki_search(question)
    
    # 2) VectorStore에 저장 (선택)
    if save_to_db:
        store_wiki_results(wiki_results)
    
    # 3) VectorStore에서 유사도 검색 
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 4) LLM 호출
    prompt = f"아래 문서를 참고하여 질문에 답변해 주세요.\n\n[문서]\n{context}\n\n[질문]\n{question}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return response.content 

# 6. 테스트
if __name__ == "__main__":
    while True:
        query = input("질문을 입력하세요: ")
        if query.lower() == "exit":
            print("종료합니다.")
            break
        answer = ask_question(query)
        print("\n[답변]:", answer)
