import os
from typing import Annotated, TypedDict
from typing_extensions import List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage

# 1. 환경 변수 로드
load_dotenv()

# 2. State 정의
class State(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# 3. 모델 설정
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. 노드 함수
def call_model(state: State):
    print(f"--- 현재 리스트 크기: {len(state['messages'])} ---")

    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 5. 그래프 구축
workflow = StateGraph(State)
workflow.add_node("chatbot", call_model)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# 6. 체크포인터 설정 및 컴파일
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 7. 실행부
config = {"configurable": {"thread_id": "user_session_001"}}

def ask_question(query: str):
    print(f"\n[사용자]: {query}")
    input_data = {"messages": [HumanMessage(content=query)]}
    output = app.invoke(input_data, config)
    print(f"[AI]: {output['messages'][-1].content}")

# 명시적으로 실행 함수 호출
if __name__ == "__main__":
    print("=== 챗봇 대화를 시작합니다 (종료하려면 'exit' 또는 'quit' 입력) ===")
    
    # 세션 ID 설정 (이 세션이 유지되는 동안 대화가 기억됩니다)
    config = {"configurable": {"thread_id": "interactive_session_001"}}
    
    while True:
        # 사용자로부터 입력을 받음
        user_input = input("\n[사용자]: ")
        
        # 종료 조건 확인
        if user_input.lower() in ["exit", "quit", "종료", "q"]:
            print("대화를 종료합니다. 안녕히 가세요!")
            break
            
        # 랭그래프 실행
        try:
            # 입력 데이터를 State 형식에 맞춰 구성
            input_data = {"messages": [HumanMessage(content=user_input)]}
            
            # 그래프 호출 (이전 대화 맥락은 thread_id 기반으로 자동 복원됨)
            output = app.invoke(input_data, config)
            
            # AI의 마지막 답변 출력
            print(f"[AI]: {output['messages'][-1].content}")
            
        except Exception as e:
            print(f"에러가 발생했습니다: {e}")