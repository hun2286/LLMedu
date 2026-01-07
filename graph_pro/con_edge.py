import os
import re
import sys
import operator
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

# 모델 초기화
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
embedding_model = HuggingFaceEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
web_search_tool = DuckDuckGoSearchRun()

# DB 로드 확인
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    print(f"[오류] DB를 찾을 수 없습니다: {persist_dir}"); sys.exit()

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
    docs = retriever.invoke(state["question"])
    
    new_context = [doc.page_content for doc in docs]
    new_sources = [doc.metadata.get('source', '알 수 없음') for doc in docs]
    
    return {"context": new_context, "sources": new_sources, "retry_count": 0}

def web_search_node(state: GraphState):
    print("\n--- [Node] 정보 부족: 웹 검색 수행 중 ---")
    results = web_search_tool.invoke(state["question"])
    return {
        "context": [f"[실시간 웹 정보] {results}"],
        "sources": ["실시간 웹 검색 결과"]
    }

def generate_node(state: GraphState):
    print("\n--- [Node] 답변 생성 중 ---")
    
    all_contexts = state["context"]
    all_sources = state["sources"]
    
    context_combined = "\n\n".join([f"[{i+1}] {content}" for i, content in enumerate(all_contexts)])
    source_list_text = "\n".join([f"[{i+1}]: {name}" for i, name in enumerate(all_sources)])
    
    prompt = [
        ("system", f"""당신은 제공된 문서만을 기반으로 질문에 답하는 전문 분석가입니다.
당신의 모든 답변은 아래의 엄격한 지침을 따라야 합니다.

[핵심 지침]
1. **정보 근거:** 답변은 반드시 제공된 [문서 내용]에 근거해야 합니다. 문서에 내용이 없으면 **"정보 없음"**이라고만 답변하세요.
2. **답변 형식 (Markdown):** 답변은 **마크다운(Markdown) 형식**을 사용하여 제목(##), 목록(*), 볼드체(**) 등으로 구조화하여 작성하십시오.
3. **내용 확장:** 문서 내용이 충분할 경우, 단순 요약을 넘어 **핵심 개념의 의미, 구체적 사례, 역사적 맥락** 등을 문서에 실린 내용만 사용하여 확장 설명하십시오.
4. **금지 사항:** 문서에 근거 없는 가정, 추론, 창작, 외부 지식, '참고:', '다음과 같습니다', '설명은 생략합니다' 등의 안내 문구를 절대 포함하지 마십시오.
5. **출처 표기 (본문):** 답변 내용에 출처 표기(예: [출처: PDF 제목])를 절대 포함하지 마십시오.
6. **줄 바꿈 및 가독성:** 답변의 **가독성을 최우선**으로 확보해야 합니다. **목록 항목(*)**의 내용이 길어지더라도, 각 항목 내에서 의미 단위가 끝날 때마다 **줄 바꿈(개행)을 충분히 활용**하여 내용이 밀집되어 보이지 않도록 작성하십시오.
7. **세부 목록 분리:** 목록 항목(*)이 단락처럼 길어지지 않도록, **내용을 여러 개의 짧은 목록(*)**으로 최대한 **쪼개서** 작성하십시오.
8. **[필수] 인라인 출처 태깅 (엄수):** 답변의 각 문단 또는 **모든** 주요 정보 뒤에는 **반드시** 해당 정보를 추출한 문서의 번호([1][2][3])를 **공백 없이 개별적으로** 붙이십시오. (예시: ...내용입니다.[1][4])

[제공된 문서 번호 목록]
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
    
    final_sources_list = []
    for tag_str in used_tags:
        idx = int(tag_str) - 1
        if 0 <= idx < len(all_sources):
            final_sources_list.append(all_sources[idx])
    
    final_sources_sorted = sorted(list(set(final_sources_list)))
    final_sources_text = "\n".join([f"[출처: {s}]" for s in final_sources_sorted])

    full_response = f"{clean_body}\n\n{'-'*60}\n{final_sources_text if final_sources_sorted else '출처 정보 누락됨'}"
    
    return {"answer": full_response, "retry_count": state.get("retry_count", 0) + 1}

# ---------------------------
# 4. 조건부 엣지(Router) 로직
# ---------------------------

def grade_documents_router(state: GraphState) -> Literal["generate", "web_search"]:
    print("--- [Edge] 검색 결과 평가 중 ---")
    if not state["context"]:
        return "web_search"
    
    score_prompt = f"""질문: {state['question']}
검색된 문서(일부): {state['context'][0][:200]}...
질문에 답변할 수 있으면 'YES', 부족하면 'NO'라고만 답하세요."""
    
    res = llm.invoke(score_prompt)
    if "yes" in res.content.strip().lower():
        return "generate"
    return "web_search"

def check_quality_router(state: GraphState) -> Literal["finish", "re_generate"]:
    print("--- [Edge] 답변 품질 검수 중 ---")
    # 정보 없음이 떴거나 답변이 너무 부실한 경우 재시도 (최대 2회)
    if ("정보 없음" in state["answer"] or len(state["answer"]) < 100) and state["retry_count"] < 2:
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

# 조건부 엣지 1: 검색 후 판별
workflow.add_conditional_edges(
    "retrieve",
    grade_documents_router,
    {
        "generate": "generate",
        "web_search": "web_search"
    }
)

workflow.add_edge("web_search", "generate")

# 조건부 엣지 2: 생성 후 품질 체크 (여기서 함수 이름을 일치시켰습니다)
workflow.add_conditional_edges(
    "generate",
    check_quality_router,
    {
        "finish": END,
        "re_generate": "generate"
    }
)

app = workflow.compile()

# ---------------------------
# 6. 실행부
# ---------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LangGraph Agentic RAG 시스템 가동")
    print("="*60)

    while True:
        query = input("\n질문을 입력하세요 (exit 종료): ").strip()
        if query.lower() == "exit": break
        if not query: continue

        inputs = {"question": query, "context": [], "sources": [], "retry_count": 0}
        
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "generate":
                    print(f"\n[최종 결과물]:\n\n{value['answer']}")
        print("\n" + "-"*60)