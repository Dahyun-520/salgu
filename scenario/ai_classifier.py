from groq import Groq
import os

client = Groq(api_key="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")


def classify_place(text):

    prompt = f"""
다음 장소 이름을 가장 가까운 유형으로 하나만 골라라.

선택지:
교실, 과학실, 컴퓨터실,
도서관, 강당, 급식실, 학생휴게실

장소명: {text}

출력은 선택지 중 하나만.
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return res.choices[0].message.content.strip()