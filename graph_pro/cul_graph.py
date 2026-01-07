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