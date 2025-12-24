# langchain과 원리는 현재 똑같은 질의응답 코드(langgraph 버전) 벡터값 활용하여 답변
# 추후 node 추가를 통해 langgraph 확장성 사용 가능

import os
from typing import Annotated, TypedDict
from typing_extensions import List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

load_dotenv()

# 1. State 정의: 대화 내용과 검색된 문서를 함께 저장
class State(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    context: List[str]  # 검색된 문서 내용을 담을 공간

# 2. 벡터 DB 로드 (기존 Cultural_db 폴더 연결)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./Cultural_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 관련 문서 3개 추출

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. 노드 정의

def retrieve(state: State):
    """[노드 1] 질문과 관련된 문서를 DB에서 찾아 context에 저장"""
    last_user_message = state["messages"][-1].content
    print(f"\n[검색] '{last_user_message}' 관련 정보를 Cultural_db에서 찾는 중...")
    
    docs = retriever.invoke(last_user_message)
    # 문서의 본문 내용만 추출해서 context에 담기
    context_list = [doc.page_content for doc in docs]
    
    return {"context": context_list}

def generate(state: State):
    """[노드 2] 검색된 정보를 바탕으로 최종 답변 생성"""
    last_user_message = state["messages"][-1].content
    context = "\n\n".join(state["context"])
    
    # RAG 전용 프롬프트 구성
    prompt = f"""당신은 문화 예술 전문가입니다. 아래 제공된 참고 정보를 바탕으로 사용자의 질문에 답하세요.
    정보가 부족하다면 모른다고 답하고 억지로 지며내지 마세요.
    
    [참고 정보]
    {context}
    
    [질문]
    {last_user_message}
    """
    
    print("[생성] 답변을 생성하고 있습니다...")
    response = model.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# 4. 그래프 구축
workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# 순서: 시작 -> 검색 -> 생성 -> 끝
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile(checkpointer=MemorySaver())

# 5. 실행
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "cultural_session_1"}}
    print("=== 문화 예술 RAG 비서 시작 ===")
    
    while True:
        user_input = input("\n[사용자]: ")
        if user_input.lower() in ["q", "exit"]: break
        
        # 입력 데이터 전달
        inputs = {"messages": [HumanMessage(content=user_input)]}
        output = app.invoke(inputs, config)
        
        print(f"\n[AI]: {output['messages'][-1].content}")