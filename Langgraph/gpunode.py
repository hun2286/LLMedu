import os
import subprocess
from typing import Annotated, TypedDict, Literal
from typing_extensions import List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

# 1. State 및 툴 정의
class State(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

@tool
def check_my_gpu():
    """내 로컬 컴퓨터의 NVIDIA GPU 상태(이름, 온도, 메모리)를 확인합니다."""
    try:
        command = "nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        result = subprocess.check_output(command, shell=True, encoding='utf-8')
        name, temp, used, total = result.strip().split(', ')
        return f"GPU: {name}, 온도: {temp}도, 메모리: {used}/{total} MiB"
    except:
        return "GPU 정보를 가져올 수 없습니다."

tools = [check_my_gpu]
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# 2. 노드 정의
def call_model(state: State):
    # 히스토리 관리 (최근 5개)
    managed_messages = state["messages"][-5:]
    response = model.invoke(managed_messages)
    return {"messages": [response]}

def execute_tools(state: State):
    # 모델이 요청한 툴 실행
    last_msg = state["messages"][-1]
    results = []
    for tool_call in last_msg.tool_calls:
        res = check_my_gpu.invoke(tool_call["args"])
        results.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
    return {"messages": results}

def make_it_polite(state: State):
    # 최종 답변을 정중하게 다듬기 (툴 결과물이나 일반 답변 모두 대상)
    last_msg_content = state["messages"][-1].content
    prompt = f"다음 문장을 아주 정중하고 친절한 말투로 고쳐줘: {last_msg_content}"
    polite_res = model.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=f"{polite_res.content}")]}

# 3. 조건부 로직
def router(state: State) -> Literal["tools", "polisher"]:
    if state["messages"][-1].tool_calls:
        return "tools"
    return "polisher"

# 4. 그래프 구축
workflow = StateGraph(State)
workflow.add_node("chatbot", call_model)
workflow.add_node("tools", execute_tools)
workflow.add_node("polisher", make_it_polite)

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", router) # 툴 쓸지 교정할지 결정
workflow.add_edge("tools", "chatbot")             # 툴 쓰고 나면 다시 모델에게
workflow.add_edge("polisher", END)               # 교정 끝나면 진짜 끝

app = workflow.compile(checkpointer=MemorySaver())

# 5. 실행
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "final_boss_session"}}
    while True:
        u_input = input("\n[사용자]: ")
        if u_input.lower() in ['q', 'exit', 'quit']: break
        output = app.invoke({"messages": [HumanMessage(content=u_input)]}, config)
        print(f"\n[AI]: {output['messages'][-1].content}")