import os
import re
import sys
import operator
from datetime import datetime
from typing import Annotated, List, TypedDict, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

# 1. 환경 설정 및 로드
load_dotenv()
persist_dir = "../database/Cultural_db"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
web_search_tool = DuckDuckGoSearchRun()

if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    print(f"[알림] 로컬 DB를 찾을 수 없습니다."); retriever = None

# 2. 그래프 상태(State) 정의
class GraphState(TypedDict):
    question: str
    context: Annotated[List[str], operator.add]
    sources: Annotated[List[str], operator.add]
    answer: str
    retry_count: int

# 3. 노드(Nodes) 정의
def retrieve_node(state: GraphState):
    print("\n--- [Node] DB 검색 수행 중 ---")
    if retriever is None:
        return {"context": [], "sources": [], "retry_count": 0}
    
    docs = retriever.invoke(state["question"])
    return {
        "context": [doc.page_content for doc in docs],
        "sources": [doc.metadata.get('source', '알 수 없음') for doc in docs],
        "retry_count": 0
    }

def web_search_node(state: GraphState):
    print("\n--- [Node] 웹 검색 최적화 및 수행 중 ---")
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # [개선] 최신성 강박 해결: 질문 성격에 따라 날짜 포함 여부 결정
    query_gen_prompt = f"""사용자 질문: {state['question']}
현재 날짜: {current_date}

위 질문에 대해 최적의 검색어 1개만 생성하세요.
- 질문이 날씨, 뉴스, 주가 등 실시간 정보나 특정 시점의 정보를 묻는 경우에만 날짜({current_date})를 포함하세요.
- 질문이 역사, 원리, 재료 등 일반적인 지식을 묻는 경우 날짜를 절대 포함하지 말고 핵심 키워드만 사용하세요.
검색어:"""
    
    search_query = llm.invoke(query_gen_prompt).content.strip().replace('"', '')
    print(f"--- [Search Query]: {search_query} ---")
    
    results = web_search_tool.invoke(search_query)
    return {
        "context": [f"[실시간 웹 정보] {results}"],
        "sources": ["웹 검색"]
    }

def generate_node(state: GraphState):
    print("\n--- [Node] 답변 생성 중 ---")
    
    all_contexts = state.get("context", [])
    context_combined = "\n\n".join(all_contexts)
    
    prompt = [
            ("system", """당신은 전문 분석가입니다. 
    1. 제공된 [데이터]를 바탕으로 사용자의 질문에 친절하게 답하십시오.
    2. 데이터에 구체적인 수치가 없더라도, 관련된 정보를 활용하여 최대한 도움이 되는 답변을 구성하십시오.
    3. 마크다운(##, *, **) 형식을 활용하여 가독성 있게 작성하십시오.
    4. 부연 설명 없이 질문에 대한 핵심 답변만 깔끔하게 구성하십시오."""), 
            ("user", f"[데이터]:\n{context_combined}\n\n질문:\n{state['question']}")
        ]
    
    response = llm.invoke(prompt)
    return {"answer": response.content.strip(), "retry_count": state.get("retry_count", 0) + 1}

# 4. 조건부 엣지(Router) 로직
def grade_documents_router(state: GraphState) -> Literal["generate", "web_search"]:
    print("--- [Edge] 문서 적합성 평가 중 ---")
    
    if not state["context"]:
        return "web_search"
    
    # DB 내용이 질문과 관련이 있는지 단순 판단
    score_prompt = f"""질문: {state['question']}\n데이터: {state['context'][0][:500]}\n
    위 데이터가 질문에 대답하는 데 직접적인 도움이 되는 내용을 포함하고 있습니까? (YES/NO)"""
    
    res = llm.invoke(score_prompt)
    if "yes" in res.content.strip().lower():
        return "generate"
    return "web_search"

def check_quality_router(state: GraphState) -> Literal["finish", "re_generate"]:
    print("--- [Edge] 품질 검수 중 ---")
    if ("정보 없음" in state["answer"] or len(state["answer"]) < 20) and state["retry_count"] < 2:
        return "re_generate"
    return "finish"

# 5. 그래프 구축
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", grade_documents_router, {"generate": "generate", "web_search": "web_search"})
workflow.add_edge("web_search", "generate")
workflow.add_conditional_edges("generate", check_quality_router, {"finish": END, "re_generate": "generate"})

app = workflow.compile()