# 히스토리 10개만 기억하고 노드 2개로 답변하는 코드

import os
from typing import Annotated, TypedDict
from typing_extensions import List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

load_dotenv()

class State(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# [노드 1] 일반 답변 생성
def call_model(state: State):
    history_limit = 10
    managed_messages = state["messages"][-history_limit:]

    print(f"\n[시스템] 전체 메시지 {len(state['messages'])}개 중 최근 {len(managed_messages)}개만 참조합니다.")
    
    response = model.invoke(managed_messages)
    print(f">>> [노드 1 결과 (교정 전)]: {response.content}")
    return {"messages": [response]}

# [노드 2] 말투 교정
def make_it_polite(state: State):
    # 바구니에서 방금 노드 1이 넣은 마지막 메시지를 꺼냅니다.
    last_ai_message = state["messages"][-1].content
    
    prompt = f"다음 문장을 아주 정중하고 친절한 말투로 수정해줘. 원래 문장의 뜻은 유지해: {last_ai_message}"
    polite_response = model.invoke([HumanMessage(content=prompt)])
    
    print(f">>> [노드 2 결과 (교정 후)]: {polite_response.content}")
    return {"messages": [AIMessage(content=polite_response.content)]}

# 그래프 구축
workflow = StateGraph(State)
workflow.add_node("chatbot", call_model)
workflow.add_node("polisher", make_it_polite)

workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", "polisher")
workflow.add_edge("polisher", END)

app = workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "comparison_session"}}
    print("=== 비교 모드 챗봇 시작 ===")
    
    while True:
        user_input = input("\n[사용자]: ")
        if user_input.lower() in ["q", "exit", "quit"]: break
        
        output = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
