import os
import re
import sys
from dotenv import load_dotenv
from typing import Optional, Union

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# 환경 변수 로드
load_dotenv()

# 설정 (DB 경로)
persist_dir = "../database/Cultural_db"

# 1. GPT-4o 초기화
print("GPT-4o 초기화 중...")
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    max_tokens=2048
)

# 2. 임베딩 모델 (기존 DB 연동)
print("임베딩 모델 로드 중...")
embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# VectorDB 로드
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    print(f"기존 DB 로드 완료: {persist_dir}")
    vectorstore = Chroma(
        persist_directory=persist_dir, 
        embedding_function=embedding_model
    )
    # 검색기 설정 (K=5)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    print(f"[오류] DB를 찾을 수 없습니다. 경로를 확인하세요: {persist_dir}")
    sys.exit()

# 지침이 반영된 프롬프트 빌더
def build_gpt_prompt(context, question, source_list_text):
    return [
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
        ("user", f"문서 내용:\n{context}\n\n질문:\n{question}")
    ]

# RAG 로직 (형태소 분석기 제거 버전)
def rag_answer(question):
    if not question.strip(): return "정보 없음"
    
    retriever_docs = retriever.invoke(question)

    context_chunks = []
    sources_map = []
    for i, doc in enumerate(retriever_docs):
        sources_map.append(doc.metadata.get('source', '알 수 없음'))
        context_chunks.append(f"[{i+1}] {doc.page_content}")

    source_list_text = "\n".join([f"[{i+1}]: {s}" for i, s in enumerate(sources_map)])
    
    messages = build_gpt_prompt("\n\n".join(context_chunks), question, source_list_text)
    
    try:
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        return f"에러 발생: {e}"

    if "정보 없음" in answer or not answer:
        return "정보 없음"

    # 1. 출처 리스트 생성을 위해 사용된 태그([1], [2] 등) 추출
    used_tags = set(re.findall(r'\[(\d+)\]', answer))
    
    # 2. [핵심 수정] 답변 본문에서 [숫자] 태그를 모두 제거 (가독성 확보)
    # 문장 끝에 붙은 [1] 혹은 문장 중간의 [2][3] 등을 모두 지웁니다.
    clean_body = re.sub(r'\[\d+\]', '', answer).strip()
    
    # 3. 실제 사용된 문서만 하단 출처 리스트에 표시
    final_sources_list = []
    for tag_str in used_tags:
        idx = int(tag_str) - 1
        if 0 <= idx < len(sources_map):
            final_sources_list.append(sources_map[idx])
    
    final_sources_sorted = sorted(list(set(final_sources_list)))
    final_sources_text = "\n".join([f"[출처: {s}]" for s in final_sources_sorted])

    # 4. 태그가 제거된 clean_body를 반환
    return f"{clean_body}\n\n{'-'*60}\n{final_sources_text if final_sources_sorted else '출처 정보 누락됨'}"

# 메인 루프
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GPT-4o RAG 시스템")
    print("="*60)

    while True:
        query = input("\n질문을 입력하세요 (exit 종료): ").strip()
        if query.lower() == "exit": break
        if not query: continue
    
        result = rag_answer(query)
        print(f"\n{result}")
        print("-"*60)