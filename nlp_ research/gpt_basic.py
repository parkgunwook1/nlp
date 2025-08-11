from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv() # .env 파일을 읽어서 환경 변수로 설정
api_key = os.getenv('OPENAI_API_KEY') # 환경 변수에서 API 키를 가져옴

if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.1,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "2022년 월드컵 우승 팀은 어디야?"},
    ]
)

print(response)

print('--------')
print(response.choices[0].message.content) # response의 내용만 출력