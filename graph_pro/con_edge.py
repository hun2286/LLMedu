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

# 판단과 생성을 위해 LLM 설정 (온도 0으로 설정하여 지침 준수 극대화)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
web_search_tool = DuckDuckGoSearchRun()

# DB 로드 확인
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    print(f"[알림] 로컬 DB를 찾을 수 없어 웹 검색 위주로 동작합니다."); retriever = None

# ---------------------------
# 2. 그래프 상태(State) 정의
# ---------------------------
class GraphState(TypedDict):
    question: str
    context: Annotated[List[str], operator.add]
    sources: Annotated[List[str], operator.add]
    answer: str
    retry_count: int

# ---------------------------
# 3. 노드(Nodes) 정의
# ---------------------------

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
    
    # [개선] 질문을 검색용 키워드로 변환
    current_date = datetime.now().strftime("%Y-%m-%d")
    query_gen_prompt = f"질문: {state['question']}\n현재날짜: {current_date}\n위 질문에 대해 최신 정보를 얻을 수 있는 검색어 1개만 출력하세요."
    search_query = llm.invoke(query_gen_prompt).content.strip()
    
    print(f"--- [Search Query]: {search_query} ---")
    results = web_search_tool.invoke(search_query)
    
    return {
        "context": [f"[실시간 웹 정보] {results}"],
        "sources": [f"실시간 웹 검색 ({search_query})"]
    }

def generate_node(state: GraphState):
    print("\n--- [Node] 답변 생성 중 ---")
    
    all_contexts = state["context"]
    all_sources = state["sources"]
    
    context_combined = "\n\n".join([f"[{i+1}] {content}" for i, content in enumerate(all_contexts)])
    source_list_text = "\n".join([f"[{i+1}]: {name}" for i, name in enumerate(all_sources)])
    
    prompt = [
        ("system", f"""당신은 전문 분석가입니다. 아래의 엄격 지침을 100% 준수하십시오.

[핵심 지침]
1. **정보 근거:** 답변은 반드시 제공된 [문서 내용] 또는 [실시간 웹 정보]에 근거해야 합니다. 단, 질문과 100% 일치하는 특정 수치가 없더라도 관련 있는 유사 정보가 있다면 이를 활용하여 최대한 답변하십시오. 관련 정보가 전무할 때만 "정보 없음"이라고 답변하세요.
2. **최신성 및 맥락 분석:** 최신 이슈는 [실시간 웹 정보]를 최우선으로 하되, 검색 결과가 파편화되어 있다면 맥락을 파악하여 사용자에게 유용한 정보로 재구성하십시오.
3. **답변 형식:** 마크다운(Markdown)을 사용하여 제목(##), 목록(*), 볼드체(**)로 구조화하십시오.
4. **가독성:** 의미 단위로 줄 바꿈을 자주 사용하고, 목록 항목을 짧게 쪼개십시오.
5. **인라인 출처:** 모든 주요 정보 뒤에 문서 번호([1][2])를 공백 없이 붙이십시오.
6. **금지 사항:** '참고:', '다음과 같습니다' 등의 사족을 절대 쓰지 마십시오.

[제공된 문서 목록]
{source_list_text}"""),
        ("user", f"문서 내용:\n{context_combined}\n\n질문:\n{state['question']}")
    ]
    
    response = llm.invoke(prompt)
    raw_answer = response.content.strip()
    
    if "정보 없음" in raw_answer or not raw_answer:
        return {"answer": "정보 없음", "retry_count": state.get("retry_count", 0) + 1}

    # 후처리: 번호 추출 및 본문 정제
    used_tags = set(re.findall(r'\[(\d+)\]', raw_answer))
    clean_body = re.sub(r'\[\d+\]', '', raw_answer).strip()
    
    # 웹 검색 결과는 하단 출처 리스트에서 제외
    final_sources_list = []
    for tag_str in used_tags:
        idx = int(tag_str) - 1
        if 0 <= idx < len(all_sources) and "웹 검색" not in all_sources[idx]:
            final_sources_list.append(all_sources[idx])
    
    final_sources_sorted = sorted(list(set(final_sources_list)))
    final_sources_text = "\n".join([f"[출처: {s}]" for s in final_sources_sorted])

    full_response = f"{clean_body}\n\n{'-'*60}\n{final_sources_text}" if final_sources_sorted else clean_body
    
    return {"answer": full_response, "retry_count": state.get("retry_count", 0) + 1}

# ---------------------------
# 4. 조건부 엣지(Router) 로직
# ---------------------------

def grade_documents_router(state: GraphState) -> Literal["generate", "web_search"]:
    print("--- [Edge] 문서 적합성 평가 중 ---")
    
    # DB 결과가 아주 많고 핵심 키워드가 겹치면 바로 생성으로 보냄
    score_prompt = f"""질문: {state['question']}
    내용: {state['context'][0][:500]}
    [판단] 
    위 내용에 질문에 대한 명확한 순서나 정의가 포함되어 있습니까?
    이미 충분한 정보가 있다면 'YES'를, 최신 정보 확인이 꼭 필요할 것 같으면 'NO'를 하세요.
    답변: """
    
    res = llm.invoke(score_prompt)
    return "generate" if "yes" in res.content.strip().lower() else "web_search"

def check_quality_router(state: GraphState) -> Literal["finish", "re_generate"]:
    print("--- [Edge] 품질 검수 중 ---")
    if "정보 없음" in state["answer"] and state["retry_count"] < 2:
        return "re_generate"
    return "finish"

# ---------------------------
# 5. 그래프 구축
# ---------------------------
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", grade_documents_router, {"generate": "generate", "web_search": "web_search"})
workflow.add_edge("web_search", "generate")
workflow.add_conditional_edges("generate", check_quality_router, {"finish": END, "re_generate": "generate"})

app = workflow.compile()

# ---------------------------
# 6. 실행
# ---------------------------
if __name__ == "__main__":
    while True:
        query = input("\n질문을 입력하세요 (exit 종료): ").strip()
        if query.lower() == "exit": break
        if not query: continue

        inputs = {"question": query, "context": [], "sources": [], "retry_count": 0}
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "generate":
                    print(f"\n[최종 답변]:\n\n{value['answer']}")