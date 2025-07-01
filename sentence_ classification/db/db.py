import pymysql
import csv

# 1. DB 연결
conn = pymysql.connect(
    host='127.0.0.1',
    port=31217,
    user='root',
    password='qwer1234',
    db='nlp',
    charset='utf8mb4'
)
cursor = conn.cursor()

# 2. SELECT 쿼리
query = "SELECT id, premise, hypothesis, label FROM nlp.insult_pair ORDER BY id"
cursor.execute(query)
rows = cursor.fetchall()

# 3. TXT 파일로 저장
with open('insult_pair_dataset.txt', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['id', 'premise', 'hypothesis', 'label'])  # 헤더
    for row in rows:
        writer.writerow(row)

# 4. 연결 종료
cursor.close()
conn.close()
print("✅ 'insult_pair_dataset_with_id.txt' 파일이 생성되었습니다.")
