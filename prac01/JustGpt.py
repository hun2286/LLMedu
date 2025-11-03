# 모델 불러와서 단순 질의응답 하는 코드
import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일에서 API 키 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

def ask_gpt(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "질문에 답변하는 역할"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def main():
    print("채팅 시작 (종료: exit)")
    while True:
        user_input = input("질문: ")
        if user_input.lower() == "exit":
            break
        answer = ask_gpt(user_input)
        print("답변:", answer)

if __name__ == "__main__":
    main()
