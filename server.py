import io
import os
import re
import cv2
import json
import httpx
import requests
import uvicorn
import numpy as np
from PIL import Image

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =============================================================
# 앱 초기화
# =============================================================

app = FastAPI(title="살구 AI 서버 (챗봇 + 구조도)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
# [챗봇] 벡터 DB & Ollama 설정
# =============================================================

INDEX_PATH = "/content/fire_index"

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "exaone3.5"

# =============================================================
# [구조도] torch.load 패치 & SAM 로드
# =============================================================

_orig_load = torch.load
def _patched_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(f, *args, **kwargs)
torch.load = _patched_load

OCR_AVAILABLE = False
ocr_reader = None

try:
    import easyocr
    ocr_reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
    print("[OCR] EasyOCR 로드 완료 (GPU:", torch.cuda.is_available(), ")")
    OCR_AVAILABLE = True
except Exception as e:
    print("[OCR] EasyOCR 비활성화:", e)

print("[SAM] Loading model...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.70,
    stability_score_thresh=0.75,
    min_mask_region_area=100,
    crop_n_layers=1,
    crop_overlap_ratio=0.3,
)

print("[SAM] Ready on:", DEVICE)


# =============================================================
# [챗봇] 스키마
# =============================================================

class Question(BaseModel):
    text: str

class Mission(BaseModel):
    title: str
    status: str

class Quiz(BaseModel):
    is_correct: bool

class FeedbackRequest(BaseModel):
    student_name: str
    missions: List[Mission]
    quizzes: List[Quiz]
    call_119: bool


# =============================================================
# [챗봇] POST /ask
# =============================================================

@app.post("/ask")
async def ask_question(query: Question):
    app_keywords = ["앱", "미션", "화면", "버튼", "챗봇", "로그인", "역할", "소화팀", "시민팀", "응급처치팀"]
    category_filter = "app_guide" if any(kw in query.text for kw in app_keywords) else "disaster"

    results = vector_db.similarity_search(query.text, k=3, filter={"category": category_filter})
    context = "\n".join([doc.page_content for doc in results])

    system_instruction = (
        "당신은 초등학생을 위한 재난 안전 시뮬레이션 '살구'의 친절한 안내 AI입니다.\n"
        "반드시 아래의 [절대 규칙]을 100% 준수하여 답변하세요.\n\n"
        "[절대 규칙]\n"
        "1. 정보 출처: 반드시 제공된 [지식 베이스]에 있는 문장만 사용하세요. "
        "지식 베이스에 없는 내용은 절대 지어내지 마세요.\n"
        "2. 출력 형식: 답변 맨 앞에 질문 내용을 한 문장으로 자연스럽게 언급한 뒤, "
        "순서가 중요한 행동 절차라면 '1. 2. 3.' 번호를 붙이고, "
        "순서가 없는 정보나 설명이라면 번호 없이 자연스러운 문장으로 작성하세요. 소제목은 달지 마세요.\n"
        "3. 말투: 초등학생을 위해 다정하고 부드러운 한국어 존댓말('~해요', '~하세요')만 사용하세요."
    )

    user_input = (
        f"질문: 소화기 어떻게 써요?\n"
        f"답변: 소화기 사용법을 알려드릴게요!\n"
        f"1. 실내에서 사용할 때는 밖으로 대피할 때를 대비해서 문을 등지고 서요.\n"
        f"2. 소화기를 가져와서 몸통을 단단히 잡고 안전핀을 뽑으세요.\n"
        f"3. 노즐을 잡고 불 쪽을 향해 가까이 이동해요.\n"
        f"4. 손잡이를 꽉 움켜쥐고 분말이 골고루 불을 덮을 수 있도록 쏘세요.\n\n"
        f"질문: 화재 날 때 엘리베이터 타면 안 돼요?\n"
        f"답변: 화재가 났을 때 엘리베이터는 절대 타면 안 돼요! "
        f"정전이 되면 엘리베이터 안에 갇힐 수 있고, 연기도 엘리베이터 통로를 타고 올라오거든요. "
        f"꼭 계단을 이용해서 대피해요.\n\n"
        f"--- 지식 베이스 ---\n{context}\n\n"
        f"질문: {query.text}\n"
        f"답변:"
    )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "system": system_instruction,
                "prompt": user_input,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "repeat_penalty": 1.1,
                    "top_p": 0.8,
                    "num_predict": 300
                }
            },
            timeout=None
        )
        llm_answer = response.json().get("response", "잠시만 기다려줘, 선생님이 생각 중이야.")

    clean_answer = (
        llm_answer
        .replace("**", "")
        .replace("_", "")
        .replace("*", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )

    return {
        "status": "success",
        "category_used": category_filter,
        "answer": clean_answer,
        "context": context
    }


# =============================================================
# [챗봇] 피드백 헬퍼
# =============================================================

def _build_llm_input(student_name: str, missions: List[dict], quizzes: List[dict], call_119: bool) -> dict:
    CALL_NAME = "119 신고"
    filtered_missions = [m for m in missions if m["title"] != CALL_NAME]

    failed: List[str] = []
    success: List[str] = []

    for m in filtered_missions:
        (success if m["status"] == "COMPLETED" else failed).append(m["title"])

    if quizzes:
        correct_count = sum(1 for q in quizzes if q["is_correct"])
        (success if correct_count >= 3 else failed).append("랜덤 퀴즈 3개 이상 맞추기")
    else:
        failed.append("랜덤 퀴즈 (데이터 없음)")

    (success if call_119 else failed).append(CALL_NAME)

    return {
        "student_name": student_name,
        "failed_missions": failed,
        "success_missions": success
    }


def _build_feedback_prompt(data: dict, mission_contexts: dict) -> str:
    name = data.get("student_name", "학생")
    given_name = name[1:] if len(name) >= 2 else name

    failed = data.get("failed_missions", [])
    success = data.get("success_missions", [])

    failed_text = ", ".join(failed) if failed else "없음"
    success_text = ", ".join(success) if success else "없음"

    success_note = (
        f"마지막에 성공한 미션 1개 골라서 {given_name}학생 칭찬하며 마무리하기"
        if success
        else "성공한 미션 없어도 포기하지 말라는 따뜻한 격려로 마무리하기"
    )

    context_block = ""
    for mission_title, context in mission_contexts.items():
        context_block += f"\n[{mission_title} 관련 지식]\n{context}\n"

    return f"""
당신은 재난 안전 교육 피드백을 제공하는 선생님 AI입니다.

[{given_name}학생 결과 요약 - 이 목록에 있는 항목만 언급하세요]
- 실패한 미션: {failed_text}
- 성공한 미션: {success_text}

[출력 형식 규칙 - 매우 중요]
- 소제목, 볼드 제목(**미션에 대해** 등), 구분선 절대 사용 금지
- 번호 목록(1. 2. 3.) 사용 금지
- 친한 선생님이 학생에게 카카오톡 메시지 보내듯 자연스러운 문단 형식으로 작성
- 이모지 1~2개만 자연스럽게 넣기
- 각 미션을 설명할 때 미션 이름을 제목처럼 앞에 따로 쓰지 말고, 문장 안에 자연스럽게 녹여서 쓸 것
- 위 요약에 없는 미션은 절대 언급하지 말 것

[말투 규칙]
- 전체 존댓말('~해요', '~하세요', '~했군요')
- 학생 이름은 맨 처음과 맨 마지막에만 "{given_name}학생!" 형식으로 사용
- 중간에는 이름 호칭 없이 자연스럽게 이어가기

[내용 규칙]
- 실패한 미션마다 아래 지식 베이스 내용을 활용해 구체적인 행동 방법 설명
- 설명이 길어질 경우 가장 긴 미션 설명 끝에 자연스럽게 "더 자세한 내용은 앱 내 메뉴얼에서 확인해 보세요!" 문구를 한 번만 녹여서 쓸 것
- {success_note}

지식 베이스:
{context_block}
"""


def _call_ollama_sync(prompt: str, num_predict: int = 500) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": num_predict,
                    "temperature": 0.3,
                    "repeat_penalty": 1.1
                }
            },
            timeout=300
        )
        if response.status_code == 200:
            return response.json().get("response", "응답 없음")
        return f"Ollama 오류: {response.status_code}"
    except Exception as e:
        return f"Ollama 연결 실패: {str(e)}"


# =============================================================
# [챗봇] 백엔드 API 기반 조회 (DB 직접 연결 대신 이걸 사용)
# =============================================================

BACKEND_API_BASE = os.environ.get(
    "BACKEND_API_BASE",
    "https://disaster-ar-backend-a7bvfvd8f6bxbsfh.koreacentral-01.azurewebsites.net"
)

CALL_NAME = "119 신고"

# evaluations 응답의 boolean 필드 -> 사람이 읽는 미션명 매핑
# ⚠️ fireteam* 3개는 소화팀 학생한테만 해당. teamCode/teamName 실제 값 확인 후
#    소화팀 아닌 학생은 이 3개를 아예 빼도록 조건 추가 필요 (확인되면 다시 고쳐드릴게요)
MISSION_LABELS = {
    "extinguisherFound": "소화기 찾기",
    "safeZoneCompleted": "안전구역으로 대피하기",
    "fireteamExtinguisherAcquired": "소화팀: 소화기 확보",
    "fireteamExtinguisherQuizCompleted": "소화팀: 소화기 사용법 퀴즈",
    "fireteamDonutCompleted": "소화팀: 도넛 미션",
}


class FeedbackAPIRequest(BaseModel):
    scenario_id: str
    student_id: str


def build_llm_input_from_evaluation(student_eval: dict) -> dict:
    failed = []
    success = []

    for field, label in MISSION_LABELS.items():
        (success if student_eval.get(field) else failed).append(label)

    (success if student_eval.get("randomQuizCompleted") else failed).append(
        "랜덤 퀴즈 3개 이상 맞추기"
    )
    (success if student_eval.get("reportCallCompleted") else failed).append(CALL_NAME)

    return {
        "student_name": student_eval.get("studentName", "학생"),
        "failed_missions": failed,
        "success_missions": success,
    }


def _generate_feedback_from_llm_input(llm_input: dict) -> str:
    mission_contexts = {}
    for mission_title in llm_input["failed_missions"]:
        results = vector_db.similarity_search(
            mission_title, k=2, filter={"category": "disaster"}
        )
        mission_contexts[mission_title] = "\n".join([doc.page_content for doc in results])

    prompt = _build_feedback_prompt(llm_input, mission_contexts)
    dynamic_tokens = min(300 + 150 * len(llm_input["failed_missions"]), 1200)
    return _call_ollama_sync(prompt, num_predict=dynamic_tokens)


def generate_feedback(student_name: str, missions: list, quizzes: list, call_119: bool) -> str:
    """기존 /feedback(JSON 직접입력) 경로용."""
    llm_input = _build_llm_input(student_name, missions, quizzes, call_119)
    return _generate_feedback_from_llm_input(llm_input)


@app.post("/feedback/from-api")
def feedback_from_api(req: FeedbackAPIRequest):
    try:
        url = f"{BACKEND_API_BASE}/api/scenarios/{req.scenario_id}/evaluations"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return JSONResponse(status_code=502, content={"detail": f"백엔드 API 조회 실패: {e}"})

    student_eval = next(
        (s for s in data.get("studentEvaluations", []) if s.get("studentId") == req.student_id),
        None
    )
    if not student_eval:
        return JSONResponse(
            status_code=404,
            content={"detail": f"studentId={req.student_id} 에 해당하는 평가 데이터를 찾을 수 없습니다."}
        )

    llm_input = build_llm_input_from_evaluation(student_eval)
    result = _generate_feedback_from_llm_input(llm_input)

    return {"result": result, "llm_input": llm_input}


# =============================================================
# [챗봇] POST /feedback (기존 - JSON 직접 입력 방식, 그대로 유지)
# =============================================================

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    result = generate_feedback(
        req.student_name,
        [m.dict() for m in req.missions],
        [q.dict() for q in req.quizzes],
        req.call_119
    )
    return {"result": result}


# =============================================================
# [구조도] 유틸 함수들
# =============================================================

def load_image(upload: UploadFile):
    try:
        data = upload.file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("[IMAGE LOAD ERROR]", e)
        return None


def preprocess_for_ocr(roi):
    h, w = roi.shape[:2]
    if w == 0 or h == 0:
        return None

    scale = max(64 / w, 64 / h, 1.0)
    scale = min(scale, 4.0)
    roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    roi = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    roi = cv2.fastNlMeansDenoisingColored(roi, None, 7, 7, 7, 21)
    return roi


def run_ocr(img, x, y, w, h):
    if not OCR_AVAILABLE or ocr_reader is None:
        return None

    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    cx = max(0, x + margin_x)
    cy = max(0, y + margin_y)
    cw = w - margin_x * 2
    ch = h - margin_y * 2

    if cw < 20 or ch < 20:
        cx, cy, cw, ch = x, y, w, h

    roi = img[cy:cy+ch, cx:cx+cw]
    if roi.size == 0:
        return None

    roi = preprocess_for_ocr(roi)
    if roi is None:
        return None

    try:
        results = ocr_reader.readtext(
            roi, detail=1, paragraph=False,
            text_threshold=0.5, low_text=0.3,
            width_ths=0.7, height_ths=0.7,
        )
        texts = []
        for (_, text, conf) in results:
            text = text.strip()
            text = re.sub(r"[^0-9A-Za-z가-힣\s\-_]", "", text).strip()
            if conf > 0.4 and len(text) > 0:
                texts.append(text)
        result = " ".join(texts)[:30] if texts else None
        if result:
            print(f"[OCR] 인식됨: '{result}' (bbox: {x},{y},{w},{h})")
        return result
    except Exception as e:
        print(f"[OCR] 오류: {e}")
        return None


def find_wall_dividers(signal, length, min_gap_ratio=0.12, wall_ratio_thresh=0.5):
    min_gap = max(int(length * min_gap_ratio), 20)
    dividers = []
    in_wall = False
    wall_start = 0

    for i, v in enumerate(signal):
        if v >= wall_ratio_thresh and not in_wall:
            in_wall = True
            wall_start = i
        elif v < wall_ratio_thresh and in_wall:
            in_wall = False
            mid = (wall_start + i) // 2
            if mid < length * 0.05 or mid > length * 0.95:
                continue
            if not dividers or mid - dividers[-1] >= min_gap:
                dividers.append(mid)

    return dividers


def split_room_horizontally(room, img, img_area):
    x, y, w, h = room["x"], room["y"], room["width"], room["height"]
    roi = img[max(0, y):y+h, max(0, x):x+w]
    if roi.size == 0:
        return [room]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ksize = max(int(h * 0.25) | 1, 21)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    _, wall_mask = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    col_ratio = wall_mask.sum(axis=0) / (h * 255.0 + 1e-6)
    dividers = find_wall_dividers(col_ratio, w, min_gap_ratio=0.12)

    if not dividers:
        return [room]

    cuts = sorted(set([0] + dividers + [w]))
    sub_rooms = []
    min_sub_area = img_area * 0.005

    for i in range(len(cuts) - 1):
        sw = cuts[i+1] - cuts[i]
        if sw < 30 or sw * h < min_sub_area:
            continue
        sub = dict(room)
        sub["id"] = f"{room['id']}-h{i}"
        sub["x"] = x + cuts[i]
        sub["width"] = sw
        sub["name"] = run_ocr(img, x + cuts[i], y, sw, h) or room["name"]
        sub_rooms.append(sub)

    if len(sub_rooms) > 1:
        print(f"[Split] '{room['name']}' → {len(sub_rooms)}개로 가로 분할")
        return sub_rooms
    return [room]


def split_room_vertically(room, img, img_area):
    x, y, w, h = room["x"], room["y"], room["width"], room["height"]
    roi = img[max(0, y):y+h, max(0, x):x+w]
    if roi.size == 0:
        return [room]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ksize = max(int(w * 0.25) | 1, 21)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    _, wall_mask = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    row_ratio = wall_mask.sum(axis=1) / (w * 255.0 + 1e-6)
    dividers = find_wall_dividers(row_ratio, h, min_gap_ratio=0.12)

    if not dividers:
        return [room]

    cuts = sorted(set([0] + dividers + [h]))
    sub_rooms = []
    min_sub_area = img_area * 0.005

    for i in range(len(cuts) - 1):
        sh = cuts[i+1] - cuts[i]
        if sh < 30 or w * sh < min_sub_area:
            continue
        sub = dict(room)
        sub["id"] = f"{room['id']}-v{i}"
        sub["y"] = y + cuts[i]
        sub["height"] = sh
        sub["name"] = run_ocr(img, x, y + cuts[i], w, sh) or room["name"]
        sub_rooms.append(sub)

    if len(sub_rooms) > 1:
        print(f"[Split] '{room['name']}' → {len(sub_rooms)}개로 세로 분할")
        return sub_rooms
    return [room]


def split_large_rooms(rooms, img, img_area):
    result = []
    for room in rooms:
        w, h = room["width"], room["height"]
        area = w * h
        ratio_wh = w / (h + 1e-6)
        ratio_hw = h / (w + 1e-6)

        if area > img_area * 0.08 and ratio_wh > 2.5:
            result.extend(split_room_horizontally(room, img, img_area))
        elif area > img_area * 0.08 and ratio_hw > 2.5:
            result.extend(split_room_vertically(room, img, img_area))
        else:
            result.append(room)
    return result


def remove_overlapping_rooms(elements):
    rooms = [e for e in elements if e["type"] == "방"]
    others = [e for e in elements if e["type"] != "방"]
    rooms.sort(key=lambda r: r["width"] * r["height"])

    kept = []
    for r1 in rooms:
        x1, y1, w1, h1 = r1["x"], r1["y"], r1["width"], r1["height"]
        area1 = w1 * h1
        skip = False

        for r2 in kept:
            x2, y2, w2, h2 = r2["x"], r2["y"], r2["width"], r2["height"]
            area2 = w2 * h2
            ix1 = max(x1, x2); iy1 = max(y1, y2)
            ix2 = min(x1+w1, x2+w2); iy2 = min(y1+h1, y2+h2)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            if inter == 0:
                continue
            union = area1 + area2 - inter
            if inter / union > 0.5 or inter / area2 > 0.9:
                skip = True
                break

        if not skip:
            kept.append(r1)

    return others + kept


def detect_rooms_by_walls(img, img_area, outline_cnt):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, blockSize=15, C=4)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    th = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8)), iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rooms = []
    room_idx = 1
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if bw < 30 or bh < 30: continue
        if area < img_area * 0.001 or area > img_area * 0.5: continue
        if bw / bh > 8 or bh / bw > 8: continue
        if cv2.pointPolygonTest(outline_cnt, (x + bw/2, y + bh/2), False) < 0: continue

        name = run_ocr(img, x, y, bw, bh) or f"room_{room_idx}"
        rooms.append({
            "id": f"auto-room-{room_idx}", "type": "방",
            "x": int(x), "y": int(y), "width": int(bw), "height": int(bh),
            "name": name, "floor": 0, "source": "Wall"
        })
        room_idx += 1
    return rooms


def run_sam_segmentation(img):
    max_size = 1024
    h, w = img.shape[:2]
    scale = 1.0

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        resized = img

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(rgb)

    if scale != 1.0:
        for m in masks:
            x, y, bw, bh = m["bbox"]
            m["bbox"] = [int(x/scale), int(y/scale), int(bw/scale), int(bh/scale)]
            m["area"] = int(m["area"] / (scale * scale))

    return masks


def analyze_floorplan(img):
    h, w = img.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    outline_cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(outline_cnt, 0.01 * cv2.arcLength(outline_cnt, True), True)

    elements = [{
        "id": "auto-outline", "type": "건물윤곽",
        "points": [{"x": int(p[0][0]), "y": int(p[0][1])} for p in approx],
        "floor": 0,
    }]

    masks = run_sam_segmentation(img)
    print("[SAM] mask count:", len(masks))

    room_idx = 1
    sam_rooms = []

    for mask in masks:
        area = mask["area"]
        x, y, bw, bh = mask["bbox"]
        if area < img_area * 0.001 or area > img_area * 0.4: continue
        if bw < 30 or bh < 30: continue
        if bw / bh > 8 or bh / bw > 8: continue
        if cv2.pointPolygonTest(outline_cnt, (x + bw/2, y + bh/2), False) < 0: continue

        corners = [(x, y), (x+bw, y), (x, y+bh), (x+bw, y+bh)]
        inside = sum(1 for px, py in corners
                     if cv2.pointPolygonTest(outline_cnt, (float(px), float(py)), False) >= 0)
        if inside < 2: continue

        name = run_ocr(img, int(x), int(y), int(bw), int(bh)) or f"room_{room_idx}"
        sam_rooms.append({
            "id": f"auto-room-{room_idx}", "type": "방",
            "x": int(x), "y": int(y), "width": int(bw), "height": int(bh),
            "name": name, "floor": 0, "source": "SAM"
        })
        room_idx += 1

    print(f"[SAM] 필터 후 방 수: {len(sam_rooms)}")
    sam_rooms = split_large_rooms(sam_rooms, img, img_area)
    print(f"[SAM] 분할 후 방 수: {len(sam_rooms)}")

    if len(sam_rooms) < 5:
        print("[Wall] SAM 부족 → 벽선 기반 탐지 실행")
        wall_rooms = detect_rooms_by_walls(img, img_area, outline_cnt)
        print(f"[Wall] 탐지된 방 수: {len(wall_rooms)}")
        elements += wall_rooms
    else:
        elements += sam_rooms

    elements = remove_overlapping_rooms(elements)
    print(f"[Final] 최종 방 수: {len([e for e in elements if e['type'] == '방'])}")
    return elements


# =============================================================
# [구조도] POST /analyze-floorplan
# =============================================================

@app.post("/analyze-floorplan")
async def analyze_floorplan_api(image: UploadFile = File(...)):
    img = load_image(image)
    if img is None:
        return JSONResponse(status_code=400, content={"detail": "이미지 로드 실패"})

    elements = analyze_floorplan(img)
    return {
        "elements": elements,
        "ocr_available": OCR_AVAILABLE,
        "sam_device": DEVICE,
    }


# =============================================================
# 헬스체크
# =============================================================

@app.get("/")
def root():
    return {"status": "ok", "sam_device": DEVICE}


# =============================================================
# 실행 진입점
# =============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)