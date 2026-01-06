import os
import re
import sys
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END

# 1. 환경 설정 및 DB 로드
load_dotenv()
persist_dir = "../database/Cultural_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    print("DB를 찾을 수 없습니다."); sys.exit()

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# 2. 그래프 상태(State) 정의
class GraphState(TypedDict):
    question: str          # 사용자 질문
    context: List[str]     # 검색된 문서 내용
    sources: List[str]     # 문서 제목(메타데이터)
    answer: str            # 최종 답변

# 3. 노드(Nodes) 정의
def retrieve_node(state: GraphState):
    """문서를 검색하여 상태에 저장합니다."""
    question = state["question"]
    docs = retriever.invoke(question)
    
    context = []
    sources = []
    for i, doc in enumerate(docs):
        sources.append(doc.metadata.get('source', '알 수 없음'))
        context.append(f"[{i+1}] {doc.page_content}")
    
    return {"context": context, "sources": sources}

def generate_node(state: GraphState):
    """지침에 따라 답변을 생성하고 후처리합니다."""
    context_text = "\n\n".join(state["context"])
    sources_map = state["sources"]
    
    source_list_text = "\n".join([f"[{i+1}]: {s}" for i, s in enumerate(sources_map)])
    
    prompt = [
        ("system", f"""당신은 제공된 문서만을 기반으로 질문에 답하는 전문 분석가입니다.
엄격 지침:
1. 근거 답변: 문서에 없으면 "정보 없음" 답변.
2. 형식: Markdown 구조화.
3. 태깅: 주요 정보 뒤에 [1][2] 등 인라인 태깅 필수.
4. 가독성: 줄 바꿈 및 세부 목록 활용 최우선.

[제공된 문서 번호 목록]
{source_list_text}"""),
        ("user", f"문서 내용:\n{context_text}\n\n질문:\n{state['question']}")
    ]
    
    response = llm.invoke(prompt)
    answer = response.content.strip()
    
    if "정보 없음" in answer or not answer:
        return {"answer": "정보 없음"}

    # 후처리: 태그 추출 및 본문 정제
    used_tags = set(re.findall(r'\[(\d+)\]', answer))
    clean_body = re.sub(r'\[\d+\]', '', answer).strip()
    
    final_sources = sorted({f"[출처: {sources_map[int(t)-1]}]" for t in used_tags if 0 < int(t) <= len(sources_map)})
    
    full_response = f"{clean_body}\n\n{'-'*60}\n" + "\n".join(final_sources)
    return {"answer": full_response}

# 4. 그래프 구축
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# 연결
workflow.set_entry_point("retrieve")      # 시작: 검색
workflow.add_edge("retrieve", "generate") # 검색 -> 생성
workflow.add_edge("generate", END)        # 생성 -> 종료

# 컴파일
app = workflow.compile()

# 5. 실행 루프
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LangGraph 기반 RAG 시스템")
    print("="*60)

    while True:
        query = input("\n질문을 입력하세요 (exit 종료): ").strip()
        if query.lower() == "exit": break
        
        # 그래프 실행
        inputs = {"question": query}
        # stream 모드를 사용하면 과정별 출력을 볼 수 있습니다.
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "generate":
                    print(f"\n[답변]:\n{value['answer']}")
        print("-"*60)