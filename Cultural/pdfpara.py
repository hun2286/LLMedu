# 출처 중복 제거, 임베딩 병렬처리로 속도 상승 cpu라서 딱히 속도 증가 없음 

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage, HumanMessage
import fitz
import torch
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed

# 환경 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

pdf_folder = r"C:\Users\BGR_NC_2_NOTE\Desktop\pdfs\20151103"
persist_dir = "./pdf_chroma_db_v3"

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=6000, openai_api_key=api_key)

# HuggingFace 임베딩 모델 로드
model_name = "bespin-global/klue-sroberta-base-continue-learning-by-mnr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 병렬 임베딩 함수
def embed_batch(documents):
    texts = [doc.page_content for doc in documents]
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    # Chroma에서 기대하는 list of vectors 형태로 변환
    return [e.tolist() for e in embeddings]

# PDF → 마크다운 변환
def pdf_to_markdown(pdf_path):
    md_text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for i, page in enumerate(pdf):
                blocks = page.get_text("blocks")
                blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
                md_text += f"# Page {i+1}\n\n"
                for block in blocks:
                    text = block[4].strip()
                    if not text:
                        continue
                    num_words = len(text.split())
                    num_lines = text.count("\n") + 1
                    if num_words <= 5 and num_lines <= 2:
                        md_text += f"# {text}\n\n"
                    elif num_words <= 15:
                        md_text += f"## {text}\n\n"
                    else:
                        md_text += f"{text}\n\n"
        return md_text
    except Exception as e:
        print(f"[오류] PDF 마크다운 변환 실패: {pdf_path}\n{e}")
        return ""

# PDF 문서 안전 로드
def load_pdf_safe(pdf_path):
    md_text = pdf_to_markdown(pdf_path)
    if md_text:
        return [Document(page_content=md_text,
                         metadata={"source": os.path.splitext(os.path.basename(pdf_path))[0]})]
    return []

# 폴더 내 모든 PDF 로드
def load_all_pdfs(pdf_folder):
    all_docs = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        docs = load_pdf_safe(pdf_path)
        all_docs.extend(docs)
    return all_docs

# 벡터 DB 생성 / 병렬 임베딩 적용
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    print("기존 벡터 DB를 로드")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=None)
else:
    print("벡터 DB 생성")
    docs = load_all_pdfs(pdf_folder)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    split_docs = text_splitter.split_documents(docs)

    batch_size = 32
    all_embeddings = []

    # 배치 생성
    def batch_generator():
        for i in range(0, len(split_docs), batch_size):
            yield split_docs[i:i+batch_size]

    # ThreadPoolExecutor로 병렬 임베딩 수행
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(embed_batch, batch) for batch in batch_generator()]
        for future in as_completed(futures):
            all_embeddings.extend(future.result())

    # Chroma에 documents와 embeddings 같이 넣기
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=all_embeddings,
        persist_directory=persist_dir
    )

print("벡터 DB 생성 완료")

# Retriever 생성
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# RAG 질문-답변
def rag_answer(question):
    retriever_docs = retriever.invoke(question)
    if isinstance(retriever_docs, Document):
        retriever_docs = [retriever_docs]

    # 동일 출처 문서 합치기
    source_dict = {}
    for doc in retriever_docs:
        source = doc.metadata.get("source", "출처 없음")
        if source not in source_dict:
            source_dict[source] = []
        source_dict[source].append(doc.page_content)

    context_texts = []
    for source, contents in source_dict.items():
        combined_content = "\n\n".join(contents)
        context_texts.append(f"[{source}]\n{combined_content}")
        
    context = "\n\n".join(context_texts)

    messages = [
        SystemMessage(content="""
            당신은 여러 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
            - 질문에 등장하는 단어를 정확히 구분하여 각각 답변하세요.
            - 문서에 없으면 '정보 없음'으로 표시하세요.
            - 가능한 한 문서 내 문맥과 키워드에 기반하여 정확하게 답변하세요.
            - 각 항목의 출처는 반드시 별도의 줄에 '[출처: ...]' 형태로 표시하세요.
            - 각 항목과 출처 사이에는 빈 줄 한 줄을 넣으세요.
            """),
        HumanMessage(content=f"문서 내용:\n{context}\n\n질문:\n{question}")
    ]

    response = llm.invoke(messages)
    return response.content

# 실행
if __name__ == "__main__":
    while True:
        query = input("질문을 입력하세요 (exit 입력 시 종료): ")
        if query.lower() == "exit":
            print("프로그램 종료")
            break

        answer = rag_answer(query)
        print("\n[답변]:\n", answer)
        print("-" * 50)
