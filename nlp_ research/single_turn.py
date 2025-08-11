from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일을 읽어서 환경 변수로 설정
api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키를 가져옴

if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key) # 오픈AI 클라이언트의 인스턴스 생성

while True:
    user_input = input("사용자")

    if user_input.lower() in ['exit', 'quit']:
        print("프로그램을 종료합니다.")
        break
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.9,
        messages=[
            {"role": "system", "content": "너는 사용자를 도와주는 상담사야"},
            {"role": "user", "content": user_input},
        ],
    )
    print('AI :' + response.choices[0].message.content)

# 위와같이 실행하면 사용자가 입력한 내용으로 API로 요청을 보내고 , response를 출력하는 챗봇이 된다.
# 허나 single_turn임으로 몇 초 전에 나눈 대화도 기억하지 못한다.

# 아래에 수정해서 다시 작성

def get_ai_response(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.9,
        messages=messages,
    )
    return response.choices[0].message.content

messages = [
    {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
]

while True:
    user_input = input("사용자")

    if user_input.lower() in ['exit', 'quit']:
        print("프로그램을 종료합니다.")
        break

    messages.append({"role": "user", "content": user_input})
    ai_response = get_ai_response(messages)
    messages.append({"role": "assistant", "content": ai_response})
    print('AI :' + ai_response)