import io
import os
import re
import cv2
import json
import math
import heapq
import httpx
import requests
import uvicorn
import numpy as np
from PIL import Image
from typing import Any, Optional

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

app = FastAPI(title="살구 AI 서버 (챗봇 + 구조도 + 네비게이션)")

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
    app_keywords = [
    "앱", "미션", "화면", "버튼", "역할",
    "소화팀", "시민팀", "응급처치팀",
    "인벤토리", "아이템", "지도", "위치", "비콘", "맵",
    "119", "신고", "전화",
    "퀴즈", "소화기", "스캔", "카메라",
]
    category_filter = "app_guide" if any(kw in query.text for kw in app_keywords) else "disaster"

    results = vector_db.similarity_search(query.text, k=3, filter={"category": category_filter})
    context = "\n".join([doc.page_content for doc in results])

    system_instruction = (
        "당신은 초등학생을 위한 재난 안전 시뮬레이션 '살구'의 친절한 안내 AI입니다.\n"
        "반드시 아래의 [절대 규칙]을 100% 준수하여 답변하세요.\n\n"
        "[절대 규칙]\n"
        "1. 정보 출처: 반드시 제공된 [지식 베이스]에 있는 내용만 사용하세요. "
        "지식 베이스에 없는 내용은 절대 지어내지 마세요. "
        "단, 지식 베이스 문장이 반말체(예: '~한다', '~해야 한다')로 되어 있어도 "
        "그대로 옮기지 말고 내용만 가져와서 반드시 존댓말로 바꿔서 답변하세요.\n"
        "2. 출력 형식: 답변 맨 앞에 질문 내용을 한 문장으로 자연스럽게 언급한 뒤, "
        "순서가 중요한 행동 절차라면 '1. 2. 3.' 번호를 붙이고, "
        "순서가 없는 정보나 설명이라면 번호 없이 자연스러운 문장으로 작성하세요. 소제목은 달지 마세요.\n"
        "3. 말투: 초등학생을 위해 다정하고 부드러운 한국어 존댓말('~해요', '~하세요')만 사용하세요. "
        "반말('~한다', '~해야 한다', '~이다' 등)은 절대 사용하지 마세요.\n"
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
                },
                "keep_alive": "60m"
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

# 완료로 인정하는 status 값들 (백엔드가 영문/한글 섞어서 보낼 수 있어서 둘 다 지원)
SUCCESS_STATUS_VALUES = {"COMPLETED", "완료", "SUCCESS", "성공"}


def _build_llm_input(student_name: str, missions: List[dict], quizzes: List[dict], call_119: bool) -> dict:
    CALL_NAME = "119 신고"
    filtered_missions = [m for m in missions if m["title"] != CALL_NAME]

    failed: List[str] = []
    success: List[str] = []

    for m in filtered_missions:
        (success if m["status"] in SUCCESS_STATUS_VALUES else failed).append(m["title"])

    quiz_total = len(quizzes)
    quiz_correct = sum(1 for q in quizzes if q.get("is_correct")) if quizzes else 0

    (success if call_119 else failed).append(CALL_NAME)

    return {
        "student_name": student_name,
        "failed_missions": failed,
        "success_missions": success,
        "quiz_correct": quiz_correct,
        "quiz_total": quiz_total,
    }


def _build_feedback_prompt(data: dict, mission_contexts: dict) -> str:
    name = data.get("student_name", "학생")

    failed = data.get("failed_missions", [])
    success = data.get("success_missions", [])
    quiz_correct = data.get("quiz_correct", 0)
    quiz_total = data.get("quiz_total", 0)

    failed_text = ", ".join(failed) if failed else "없음"
    success_text = ", ".join(success) if success else "없음"

    if quiz_total > 0:
        quiz_line = f"퀴즈: {quiz_total}개 중 {quiz_correct}개 정답"
    else:
        quiz_line = "퀴즈: 응시 기록 없음 (퀴즈에 대해 아무 말도 하지 말 것)"

    context_block = ""
    for mission_title, context in mission_contexts.items():
        context_block += f"[{mission_title}] {context}\n"

    return f"""너는 재난 안전 교육 피드백을 주는 선생님이다. {name} 학생에게 짧고 다정한 존댓말로 피드백을 써라.

실패: {failed_text}
성공: {success_text}
{quiz_line}
참고지식:
{context_block}

규칙: 소제목/번호/별표 사용 금지, 3~5문장 이내로 간결하게. 위에 나열되지 않은 미션·역할·오답 내용·개념은 절대 지어내지 말 것. 실패 항목은 참고지식을 활용해 방법을 짧게 설명하고, 성공 항목이 있으면 1개만 골라 칭찬. 퀴즈는 성공/실패로 나누지 말고 정답 개수 그대로 비율에 맞게 자연스럽게 코멘트. 이름은 맨 처음에만 "{name} 학생!" 형식으로 한 번만 사용."""


def _call_ollama_sync(prompt: str, num_predict: int = 350) -> str:
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
                },
                "keep_alive": "60m"
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
    # 속도 개선: 프롬프트가 3~5문장으로 짧게 나오도록 지시했으므로 토큰 상한도 낮춤
    dynamic_tokens = min(150 + 60 * len(llm_input["failed_missions"]), 500)
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
# [네비게이션] A* 경로탐색 - navigation_astar.py 이식
# GPU 미사용 (순수 CPU 연산), Ollama/SAM 자원과 경쟁하지 않음
# =============================================================

CORRIDOR_STEP: float = 35.0
WP_MAX_DIST: float = CORRIDOR_STEP * 1.45
CONN_DIST: float = 120.0
DOOR_MARGIN: float = 10.0
STAIR_COST: float = 150.0

ZONE_WEIGHTS = {"safe": 1.0, "normal": 1.1, "restricted": 8.0, "danger": 50.0}


def pt_in_room(x, y, rooms, margin=1):
    for r in rooms:
        if (r["x"] + margin < x < r["x"] + r["w"] - margin and
                r["y"] + margin < y < r["y"] + r["h"] - margin):
            return r
    return None


def seg_blocked(x1, y1, x2, y2, rooms, samples=16):
    for i in range(1, samples):
        t = i / samples
        if pt_in_room(x1 + t*(x2-x1), y1 + t*(y2-y1), rooms, margin=1):
            return True
    return False


def door_exit_point(dx, dy, rooms):
    """문 위치에서 복도 방향 좌표 + 소속 방 id"""

    best_r = None

    BOUNDARY = 20.0
    candidates = []

    for r in rooms:

        on_top = (
            abs(dy - r["y"]) < BOUNDARY
            and r["x"] - 5 <= dx <= r["x"] + r["w"] + 5
        )

        on_bottom = (
            abs(dy - (r["y"] + r["h"])) < BOUNDARY
            and r["x"] - 5 <= dx <= r["x"] + r["w"] + 5
        )

        on_left = (
            abs(dx - r["x"]) < BOUNDARY
            and r["y"] - 5 <= dy <= r["y"] + r["h"] + 5
        )

        on_right = (
            abs(dx - (r["x"] + r["w"])) < BOUNDARY
            and r["y"] - 5 <= dy <= r["y"] + r["h"] + 5
        )

        if on_top or on_bottom or on_left or on_right:

            dist = math.hypot(
                dx - (r["x"] + r["w"] / 2),
                dy - (r["y"] + r["h"] / 2)
            )

            candidates.append((dist, r))

    if candidates:

        special = [
            c for c in candidates
            if c[1]["zone"] in ("safe", "danger")
        ]

        if special:
            best_r = min(
                special,
                key=lambda x: x[0]
            )[1]
        else:
            best_r = min(
                candidates,
                key=lambda x: x[0]
            )[1]

    if not best_r:
        return dx, dy, None

    r = best_r

    faces = {
        "top": (
            abs(dy - r["y"]),
            dx,
            r["y"] - DOOR_MARGIN
        ),

        "bottom": (
            abs(dy - (r["y"] + r["h"])),
            dx,
            r["y"] + r["h"] + DOOR_MARGIN
        ),

        "left": (
            abs(dx - r["x"]),
            r["x"] - DOOR_MARGIN,
            dy
        ),

        "right": (
            abs(dx - (r["x"] + r["w"])),
            r["x"] + r["w"] + DOOR_MARGIN,
            dy
        ),
    }

    _, cx, cy = min(
        faces.values(),
        key=lambda v: v[0]
    )

    return cx, cy, r["id"]


def build_floor_graph(fd: dict) -> "tuple[dict, list]":
    """fd: {rooms, doors, beacons, bbox, floor_index}
    (비상구는 더 이상 사용하지 않음 — 안전구역(zone=safe)이 그 역할을 대신함)

    계단(is_stair=True)은 다른 방들과 다르게 취급한다:
    - 문(door)이 없어도 됨 → 복도 노드와 거리 기반으로 바로 연결
    - "방은 통과 불가" 장애물 규칙 대상이 아님 → 복도 격자 생성/직선 시야 차단
      판정(pt_in_room, seg_blocked)에서 계단은 제외하고, 사실상 뻥 뚫린
      복도의 일부처럼 취급한다.
    """
    rooms = fd["rooms"]
    doors = fd["doors"]
    fi    = fd["floor_index"]
    bb    = fd["bbox"]

    # 장애물 판정용 방 목록 — 계단은 여기서 제외 (통과 가능하므로 막지 않음)
    blocking_rooms = [r for r in rooms if not r.get("is_stair")]

    nodes: dict = {}

    # ── 복도 격자 WP (계단 영역도 막힘 없이 격자가 그대로 깔림) ──
    x0, y0, x1, y1 = bb["x0"], bb["y0"], bb["x1"], bb["y1"]
    gx = x0 + CORRIDOR_STEP / 2
    while gx <= x1:
        gy = y0 + CORRIDOR_STEP / 2
        while gy <= y1:
            if not pt_in_room(gx, gy, blocking_rooms):
                nid = f"wp_{fi}_{int(gx)}_{int(gy)}"
                nodes[nid] = {"x": gx, "y": gy, "zone": "normal",
                               "ntype": "waypoint", "room_id": None, "floor": fi}
            gy += CORRIDOR_STEP
        gx += CORRIDOR_STEP

    # ── 문 복도 포인트 (일반 방용 — 계단은 애초에 문이 없음) ──
    for door in doors:

        cx, cy, rid = door_exit_point(
            door["x"],
            door["y"],
            blocking_rooms
        )

        # 안전구역/재난구역은 가장 가까운 방으로 강제 매칭
        nearest_special = None
        nearest_dist = float("inf")

        for r in blocking_rooms:
            if r["zone"] in ("safe", "danger"):
                dist = math.hypot(
                    door["x"] - (r["x"] + r["w"]/2),
                    door["y"] - (r["y"] + r["h"]/2)
                )

                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_special = r

        if nearest_special:
            rid = nearest_special["id"]


        if pt_in_room(cx, cy, blocking_rooms):
            for off in [20, 35, 50]:
                found = False

                for ddx, ddy in [
                    (off,0),
                    (-off,0),
                    (0,off),
                    (0,-off)
                ]:
                    if not pt_in_room(
                        door["x"]+ddx,
                        door["y"]+ddy,
                        blocking_rooms
                    ):
                        cx = door["x"] + ddx
                        cy = door["y"] + ddy
                        found = True
                        break

                if found:
                    break


        rid_with_floor = f"{rid}_fl{fi}" if rid else None

        nid = f"door_{door['id']}_fl{fi}"

        nodes[nid] = {
            "x": cx,
            "y": cy,
            "zone": "normal",
            "ntype": "door",
            "room_id": rid_with_floor,
            "floor": fi,
            "door_x": door["x"],
            "door_y": door["y"]
        }

    # ── 방 중심 (목적지/출발지 전용, 계단 방도 여기 포함됨) ──
    for r in rooms:
        # nid에 floor 포함 — 같은 element_id가 다른 층에 있어도 구분
        nid = f"room_{r['id']}_fl{fi}"
        nodes[nid] = {
            "x": r["x"] + r["w"]/2, "y": r["y"] + r["h"]/2,
            "zone": r["zone"], "ntype": "room",
            "room_id": f"{r['id']}_fl{fi}",
            "floor": fi, "room_name": r["name"],
            "is_stair": r.get("is_stair", False),
        }

    # ── 간선 ──
    edges: list = []
    nl = list(nodes.items())

    for i, (aid, na) in enumerate(nl):
        for bid, nb in nl[i+1:]:
            ax, ay = na["x"], na["y"]
            bx, by = nb["x"], nb["y"]
            dist   = math.hypot(ax-bx, ay-by)
            ta, tb = na["ntype"], nb["ntype"]

            if ta == "room" or tb == "room":
                rn = na if ta == "room" else nb
                dn = nb if ta == "room" else na

                if rn.get("is_stair"):
                    # 계단: 문 필요 없이, 주변 복도/문 노드와 거리 기반으로 연결.
                    # 장애물 판정도 blocking_rooms(계단 제외)로 하므로 자기 자신이
                    # 시야를 막지 않는다 — rooms_excl 처리 불필요.
                    if dn["ntype"] in ("waypoint", "door"):
                        if dist > CONN_DIST:
                            continue
                        if seg_blocked(ax, ay, bx, by, blocking_rooms):
                            continue
                        cost = dist * max(ZONE_WEIGHTS.get(na["zone"], 1.1),
                                          ZONE_WEIGHTS.get(nb["zone"], 1.1))
                        edges.append((aid, bid, cost))
                    continue

                # 일반 방: 방 중심 ↔ 문(같은 방)만 연결
                if dn["ntype"] == "door" and dn.get("room_id") == rn["room_id"]:
                    own_id = rn["room_id"]
                    rooms_excl = [r for r in blocking_rooms if f"{r['id']}_fl{fi}" != own_id]
                    if not seg_blocked(rn["x"], rn["y"], dn["x"], dn["y"], rooms_excl):
                        edges.append((aid, bid, dist))
                continue

            # WP ↔ WP: 격자 이웃만
            if ta == "waypoint" and tb == "waypoint":
                if dist > WP_MAX_DIST: continue
            elif ta == "waypoint" or tb == "waypoint":
                if dist > CONN_DIST: continue
            else:
                if dist > CONN_DIST: continue

            if seg_blocked(ax, ay, bx, by, blocking_rooms): continue

            cost = dist * max(ZONE_WEIGHTS.get(na["zone"], 1.1),
                              ZONE_WEIGHTS.get(nb["zone"], 1.1))
            edges.append((aid, bid, cost))

    return nodes, edges


def build_all_graphs(floor_data: list) -> "tuple[dict, dict]":
    """
    계단은 별도 좌표 입력 없이, '이름에 "계단"이 들어간 방'을 자동으로 찾아
    인접 층의 계단 방들끼리 좌표 근접(최근접) 매칭으로 자동 연결한다.

    이름은 매칭에 쓰지 않는다 — 같은 건물에서 계단이 여러 개(동/서 등)여도
    전부 "계단"이라는 동일한 이름을 쓰는 경우가 많기 때문에, 이름 대신
    방 중심 좌표(x, y)가 가장 가까운 것끼리 짝짓는다. 층마다 배치가 거의
    그대로 이어지는 건물 구조상, 진짜 같은 계단실은 좌표가 가장 가깝다는
    전제.
    """
    all_nodes: dict = {}
    all_edges: list = []

    # floor_index -> [(room_node_id, cx, cy), ...] — 계단으로 인식된 방들만 모음
    stair_rooms_by_floor: dict = {}

    for fd in floor_data:
        fi = fd["floor_index"]
        lnodes, ledges = build_floor_graph(fd)
        all_nodes.update(lnodes)
        all_edges.extend(ledges)

        for r in fd["rooms"]:
            if r.get("is_stair"):
                node_id = f"room_{r['id']}_fl{fi}"
                cx = r["x"] + r["w"] / 2
                cy = r["y"] + r["h"] / 2
                stair_rooms_by_floor.setdefault(fi, []).append((node_id, cx, cy))

    # 인접한 층(fi, fi+1)끼리 계단 방을 좌표 최근접으로 그리디 매칭
    floors_sorted = sorted(stair_rooms_by_floor.keys())
    for fi in floors_sorted:
        fi_next = fi + 1
        if fi_next not in stair_rooms_by_floor:
            continue  # 바로 위층이 없으면(중간층 건너뛰기 방지) 매칭하지 않음

        lower = stair_rooms_by_floor[fi]
        upper = stair_rooms_by_floor[fi_next]

        # 가능한 모든 (lower, upper) 쌍을 거리순으로 정렬 후 그리디로 짝짓기
        candidates = []
        for li, (lid, lx, ly) in enumerate(lower):
            for ui, (uid, ux, uy) in enumerate(upper):
                dist = math.hypot(lx - ux, ly - uy)
                candidates.append((dist, li, ui, lid, uid))
        candidates.sort(key=lambda c: c[0])

        used_lower: set = set()
        used_upper: set = set()
        for dist, li, ui, lid, uid in candidates:
            if li in used_lower or ui in used_upper:
                continue
            used_lower.add(li)
            used_upper.add(ui)
            all_edges.append((lid, uid, STAIR_COST))
            all_edges.append((uid, lid, STAIR_COST))

            print(
                "[STAIR CONNECT]",
                fi, "층", lid,
                "<->",
                fi_next, "층", uid,
                "distance=", dist
            )

        # 짝을 못 찾은 계단(층별 개수가 다른 경우)은 로그로 남김 — 고립 노드가 됨
        unmatched_lower = [lower[i][0] for i in range(len(lower)) if i not in used_lower]
        unmatched_upper = [upper[i][0] for i in range(len(upper)) if i not in used_upper]
        if unmatched_lower or unmatched_upper:
            print(f"[STAIR MATCH] {fi}층↔{fi_next}층: 짝 못 찾은 계단 - "
                  f"{fi}층: {unmatched_lower}, {fi_next}층: {unmatched_upper}")

    graph: dict = {}
    for a, b, cost in all_edges:
        graph.setdefault(a, []).append((b, cost))
        graph.setdefault(b, []).append((a, cost))

    print("===== STAIR GRAPH CONNECTION =====")
    for nid, node in all_nodes.items():
        if node.get("is_stair"):
            print(
                nid,
                "floor=", node["floor"],
                "name=", node.get("room_name"),
                "edges=", graph.get(nid)
            )
    print("==============================")

    return all_nodes, graph


def astar(all_nodes: dict, graph: dict, start_id: str, goal_id: str):
    if start_id not in all_nodes or goal_id not in all_nodes: return None
    if start_id == goal_id: return [start_id]
    goal = all_nodes[goal_id]

    def h(nid: str) -> float:
        n = all_nodes[nid]
        return (math.hypot(n["x"]-goal["x"], n["y"]-goal["y"])
                + abs(n["floor"]-goal["floor"]) * 50)

    g: dict = {start_id: 0.0}
    came: dict = {start_id: None}
    pq   = [(h(start_id), start_id)]
    vis: set = set()

    while pq:
        _, cur = heapq.heappop(pq)
        if cur in vis: continue
        vis.add(cur)
        if cur == goal_id:
            path: list = []
            nd = goal_id
            while nd is not None:
                path.append(nd); nd = came.get(nd)
            path.reverse(); return path
        for nb, cost in graph.get(cur, []):
            if nb in vis: continue
            tg = g[cur] + cost
            if tg < g.get(nb, math.inf):
                g[nb] = tg; came[nb] = cur
                heapq.heappush(pq, (tg + h(nb), nb))
    return None


def nearest_corridor(x: float, y: float, floor_index: int, all_nodes: dict):
    best, bd = None, math.inf
    for nid, n in all_nodes.items():
        if n.get("floor") != floor_index: continue
        if n["ntype"] == "room": continue
        d = math.hypot(n["x"]-x, n["y"]-y)
        if d < bd: bd, best = d, nid
    return best


def _floor_element_key(floor, element_id: str) -> str:
    """JSON에서도 안전하게 쓸 수 있는 층+element_id 복합 키."""
    return f"{floor}:{element_id}"


def _find_element(elements_json: list, element_id: str, floor=None, label: str = "요소") -> dict:
    """element_id를 찾되, 중복 ID이면 floor 없이 임의 선택하지 않는다."""
    matches = [el for el in elements_json if el.get("id") == element_id]

    if floor is not None:
        matches = [el for el in matches if el.get("floor", 0) == floor]
        if not matches:
            raise ValueError(
                f"{label}를 찾을 수 없습니다: floor={floor}, element_id={element_id}"
            )
        return matches[0]

    if not matches:
        raise ValueError(f"{label}를 찾을 수 없습니다: element_id={element_id}")

    if len(matches) > 1:
        floors = sorted({el.get("floor", 0) for el in matches}, key=str)
        raise ValueError(
            f"같은 element_id가 여러 층에 존재합니다. {label}의 floor를 지정해야 합니다: "
            f"element_id={element_id}, floors={floors}"
        )

    return matches[0]


def _lookup_tags(tags_map: dict, floor, element_id: str, duplicate_ids: set) -> dict:
    """
    floor-aware tags_map과 기존 element_id-only tags_map을 모두 지원한다.

    권장 키: "{floor}:{element_id}"
    기존 키가 element_id 하나뿐이고 그 ID가 여러 층에 중복되면,
    잘못된 층으로 태그가 번지는 것을 막기 위해 해당 태그는 적용하지 않는다.
    """
    if not isinstance(tags_map, dict):
        return {}

    # 신규 JSON-safe 복합 키
    for key in (
        _floor_element_key(floor, element_id),
        f"{floor}|{element_id}",
        f"{floor}/{element_id}",
    ):
        value = tags_map.get(key)
        if isinstance(value, dict):
            return value

    # 선택적으로 {"0": {"element-id": {...}}} 형태도 지원
    floor_bucket = tags_map.get(str(floor), tags_map.get(floor))
    if isinstance(floor_bucket, dict):
        value = floor_bucket.get(element_id)
        if isinstance(value, dict):
            return value

    # 구버전 {element_id: {...}}는 ID가 유일할 때만 안전하게 사용
    value = tags_map.get(element_id)
    if element_id not in duplicate_ids and isinstance(value, dict):
        return value

    return {}


def _ref_floor_and_id(ref):
    """ElementRef / dict / legacy string을 (floor, element_id)로 정규화."""
    if isinstance(ref, str):
        return None, ref
    if isinstance(ref, dict):
        return ref.get("floor"), ref.get("element_id") or ref.get("id")
    return getattr(ref, "floor", None), getattr(ref, "element_id", None)


def parse_elements(elements_json: list, tags_map: dict, outline_bboxes: dict = None) -> list:
    """
    elements_json + tags_map → floor_data 목록
    floor_data[i] = {floor_index, rooms, doors, beacons, bbox}

    - 비상구는 더 이상 별도 타입으로 취급하지 않음 (안전구역이 대피 목적지 역할)
    - 계단은 별도 element 타입이 아니라, 이름에 "계단"이 포함된 "방"으로 저장됨
      (예: name="동쪽 계단") → is_stair=True로 표시해서 build_all_graphs에서 자동으로
      같은 이름의 계단 방을 인접 층끼리 연결함
    """
    TYPE_MAP = {
    "SAFE_ZONE": "안전 구역",
    "DISASTER_ZONE": "재난 구역",
    "RESTRICTED_ZONE": "제한 구역",
    } 

    ZONE_MAP = {
        "방": "normal",
        "안전 구역": "safe", "SAFE_ZONE": "safe",
        "재난 구역": "danger", "DANGER_ZONE": "danger",
        "제한 구역": "restricted", "RESTRICTED_ZONE": "restricted",
    }
    STAIR_KEYWORD = "계단"
    by_floor: dict = {}

    id_counts: dict = {}
    for _el in elements_json:
        _eid = _el.get("id")
        if _eid is not None:
            id_counts[_eid] = id_counts.get(_eid, 0) + 1
    duplicate_ids = {eid for eid, count in id_counts.items() if count > 1}

    def get_floor(fi: int) -> dict:
        if fi not in by_floor:
            by_floor[fi] = {"floor_index": fi,
                             "rooms": [], "doors": [], "beacons": []}
        return by_floor[fi]

    for el in elements_json:
        raw_type = str(el.get("type", "")).strip()
        zone_type = str(el.get("zoneType", "")).strip()
        element_type = str(el.get("elementType", "")).strip()

        types = [raw_type, zone_type, element_type]

        t = None

        for typ in types:
            if typ in TYPE_MAP:
                t = TYPE_MAP[typ]
                break
            elif typ in ("방", "안전 구역", "재난 구역", "제한 구역", "문", "비콘"):
                t = typ
                break

        if t is None:
            t = raw_type
            print(
                "[TYPE CHECK]",
                el.get("id"),
                raw_type,
                zone_type,
                element_type,
                "=>",
                t
            )

        eid = el["id"]
        floor = el.get("floor", 0)

        tags = _lookup_tags(tags_map, floor, eid, duplicate_ids)
        zone = tags.get("zone_type", ZONE_MAP.get(t, "normal"))
        fi = floor
        fd = get_floor(fi)

        if el["id"] == "auto-room-25":
                print(
                    "FOUND:",
                    el["id"],
                    "floor=",
                    el.get("floor"),
                    "type=",
                    el.get("type"),
                    "name=",
                    el.get("name"),
                )

        if "계단" in str(el.get("name", "")):
            print(
                "[STAIR DEBUG]",
                "floor=", fi,
                "id=", el.get("id"),
                "type=", el.get("type"),
                "name=", el.get("name")
            )

        if t in ("방", "안전 구역", "재난 구역", "제한 구역", "SAFE_ZONE", "DANGER_ZONE", "RESTRICTED_ZONE"):
            name = el.get("name", t)
            fd["rooms"].append({
                "id":   eid,
                "name": name,
                "x":    float(el.get("x", 0)),
                "y":    float(el.get("y", 0)),
                "w":    float(el.get("width", 0)),
                "h":    float(el.get("height", 0)),
                "zone": zone,
                "floor_index": fi,
                "is_stair": "계단" in el.get("name", ""),
            })
        elif t == "문":
            fd["doors"].append({
                "id": eid,
                "x":  float(el.get("x", 0)),
                "y":  float(el.get("y", 0)),
            })
        elif t == "비콘":
            fd["beacons"].append({
                "id":   eid,
                "x":    float(el.get("x", 0)),
                "y":    float(el.get("y", 0)),
                "zone": zone,
            })


    # bbox 계산
    for fi, fd in by_floor.items():
        ob = (outline_bboxes or {}).get(fi)
        if ob:
            fd["bbox"] = ob
        elif fd["rooms"]:
            x0 = min(r["x"] for r in fd["rooms"])
            y0 = min(r["y"] for r in fd["rooms"])
            x1 = max(r["x"]+r["w"] for r in fd["rooms"])
            y1 = max(r["y"]+r["h"] for r in fd["rooms"])
            fd["bbox"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        else:
            fd["bbox"] = {"x0": 0, "y0": 0, "x1": 1000, "y1": 1000}

    print("===== parse result =====")
    for fd in by_floor.values():
        print(
            fd["floor_index"],
            [r["id"] for r in fd["rooms"] if r["id"] == "auto-room-25"]
        )

    return sorted(by_floor.values(), key=lambda fd: fd["floor_index"])


def _no_path(start: str, goal: str) -> dict:
    return {"path": [], "start_element_id": start, "goal_element_id": goal,
            "total_cost": None, "found": False, "warning": "no_path_found",
            "_all_nodes": {}, "_path_ids": []}


def compute_navigation_route(
    elements_json: "list[dict[str, Any]]",
    tags_map: "dict[str, dict[str, Any]]",
    current_beacon_element_id: str,
    target_element_id: str,
    disaster_element_refs: list = None,
    outline_bboxes: dict = None,
    target_node_id: str = None,
    current_beacon_floor=None,
    target_floor=None,
    disaster_element_ids: list = None,
) -> "dict[str, Any]":
    """
    층별 element_id 중복을 안전하게 처리하는 경로 계산 함수.

    신규 방식에서는 current_beacon_floor / target_floor와
    disaster_element_refs=[{floor, element_id}, ...]를 전달한다.
    구버전 element_id-only 요청도 ID가 전체 지도에서 유일한 경우에는 호환된다.
    """
    # tags_map은 JSON-safe 복합키("floor:element_id")와 기존 ID-only 키를 모두 지원한다.
    eff_tags: dict = {}
    for key, value in (tags_map or {}).items():
        eff_tags[key] = dict(value) if isinstance(value, dict) else value

    # 구버전 disaster_element_ids도 refs로 합쳐서 처리
    refs = list(disaster_element_refs or [])
    refs.extend(disaster_element_ids or [])

    # 재난구역은 반드시 실제 층을 확정한 후 floor-aware 키로 오버라이드
    for ref in refs:
        dis_floor, dis_id = _ref_floor_and_id(ref)
        if not dis_id:
            raise ValueError("재난구역 element_id가 비어 있습니다.")

        dis_el = _find_element(
            elements_json,
            dis_id,
            dis_floor,
            label="재난구역",
        )
        resolved_floor = dis_el.get("floor", 0)
        key = _floor_element_key(resolved_floor, dis_id)
        value = eff_tags.get(key)
        if not isinstance(value, dict):
            value = {}
        else:
            value = dict(value)
        value["zone_type"] = "danger"
        value.setdefault("passable", True)
        eff_tags[key] = value

    # 파싱 (계단 방은 parse_elements에서 이름 기반으로 자동 태깅됨)
    floor_data = parse_elements(elements_json, eff_tags, outline_bboxes)

    # 그래프 구성
    all_nodes, graph = build_all_graphs(floor_data)

    print("===== STAIR ROOMS =====")
    for nid, n in all_nodes.items():
        if n.get("is_stair") is True:
            print(nid, n)
    print("==============================")

    print("===== DOOR NODE CHECK =====")
    for node_id, node in all_nodes.items():
        if node.get("ntype") == "door" and node.get("floor") == 0:
            print(node_id, node)
    print("============================")

    # 출발 비콘: floor가 있으면 정확히 (floor, id)로, 없으면 ID가 유일할 때만 허용
    bc_el = _find_element(
        elements_json,
        current_beacon_element_id,
        current_beacon_floor,
        label="출발 비콘",
    )
    start_floor = bc_el.get("floor", 0)
    start_id = nearest_corridor(
        float(bc_el.get("x", 0)),
        float(bc_el.get("y", 0)),
        start_floor,
        all_nodes,
    )

    if not start_id:
        return _no_path(str(current_beacon_element_id), target_element_id)

    # 목적지 노드
    if target_node_id and target_node_id in all_nodes:
        goal_id = target_node_id
    else:
        target_el = _find_element(
            elements_json,
            target_element_id,
            target_floor,
            label="목적지",
        )
        target_fi = target_el.get("floor", 0)
        goal_id = f"room_{target_element_id}_fl{target_fi}"

    if goal_id not in all_nodes:
        return _no_path(start_id, target_element_id)

    print("===== FINAL ROUTE CHECK =====")
    print("start_id:", start_id)
    print("goal_id:", goal_id)
    print("start exists:", start_id in all_nodes)
    print("goal exists:", goal_id in all_nodes)
    print("start edges:", graph.get(start_id))
    print("goal edges:", graph.get(goal_id))
    print("==============================")

    # A*
    path_ids = astar(all_nodes, graph, start_id, goal_id)

    print("===== ROUTE RESULT =====")
    print("found:", bool(path_ids))
    print("path length:", len(path_ids) if path_ids else 0)
    print("warning:", None if path_ids else "no_path_found")
    print("========================")
    print("PATH IDS:", path_ids)

    if not path_ids:
        return _no_path(start_id, goal_id)

    # 직렬화
    path_detail: list = []
    total = 0.0
    for i, nid in enumerate(path_ids):
        n = all_nodes[nid]
        path_detail.append({
            "element_id": nid,
            "name": n.get("room_name", n["ntype"]),
            "cx": round(n["x"], 2),
            "cy": round(n["y"], 2),
            "zone_type": n["zone"],
            "floor": n["floor"],
            "ntype": n["ntype"],
        })
        if i > 0:
            p = all_nodes[path_ids[i-1]]
            total += math.hypot(n["x"] - p["x"], n["y"] - p["y"])

    zones = {all_nodes[nid]["zone"] for nid in path_ids}
    warning = (
        "danger_zone_in_path" if "danger" in zones else
        "restricted_zone_in_path" if "restricted" in zones else
        None
    )

    return {
        "path": path_detail,
        "start_element_id": start_id,
        "goal_element_id": goal_id,
        "total_cost": round(total, 2),
        "found": True,
        "warning": warning,
        "_all_nodes": all_nodes,
        "_path_ids": path_ids,
    }


class ElementRef(BaseModel):
    # 신규 연동에서는 floor를 함께 보내는 것을 권장한다.
    # 생략 시 element_id가 전체 지도에서 유일한 경우에만 허용된다.
    floor: Optional[int] = None
    element_id: str

'''
class StairPositionRef(BaseModel):
    x: float
    y: float
    floor: int
'''

class NavigationRouteRequest(BaseModel):
    elements_json: List[dict]
    tags_map: dict = {}

    # 신규 방식: floor + element_id로 명확하게 지정 (element_id가 여러 층에서 중복될 수 있어서 추가됨)
    current_beacon: Optional[ElementRef] = None
    target: Optional[ElementRef] = None
    disaster_elements: Optional[List[ElementRef]] = None

    # 기존 호환용 (flat 방식) — 신규 필드가 없을 때만 사용됨
    current_beacon_element_id: Optional[str] = None
    target_element_id: Optional[str] = None
    disaster_element_ids: Optional[List[str]] = None
    outline_bboxes: Optional[dict] = None
    target_node_id: Optional[str] = None


@app.post("/route")
def navigation_route(req: NavigationRouteRequest):
    """elements_json/tags_map을 직접 JSON으로 넘기는 방식 (테스트 및 실제 연동용).
    계단은 별도 필드 없이, elements_json 안에 이름이 '계단'을 포함한 방으로 들어있으면 자동 인식됨.
    stair_positions로 좌표를 직접 주면, 그 좌표에 "계단" 방을 자동으로 만들어서 같은 방식으로 처리함."""

    # 🔍 디버그: 실제로 들어온 요청 body를 콘솔에 그대로 찍음 (원인 파악되면 지워도 됨)
    print("=== /route 요청 수신 ===")
    print("current_beacon:", req.current_beacon.dict() if req.current_beacon else None)
    print("current_beacon_element_id:", req.current_beacon_element_id)
    print("target:", req.target.dict() if req.target else None)
    print("target_element_id:", req.target_element_id)
    print("target_node_id:", req.target_node_id)
    print("disaster_elements:", [d.dict() for d in req.disaster_elements] if req.disaster_elements else None)
    print("disaster_element_ids:", req.disaster_element_ids)
    print("elements_json 개수:", len(req.elements_json))
    print("========================")

    try:
        elements_json = list(req.elements_json)  # 계단 좌표 주입 시 원본 안 건드리게 복사

        # 1) 출발 위치
        current_beacon_element_id = None
        current_beacon_floor = None

        if req.current_beacon:
            current_beacon_element_id = req.current_beacon.element_id
            current_beacon_floor = req.current_beacon.floor

        elif req.current_beacon_element_id:
            current_beacon_element_id = req.current_beacon_element_id

        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "current_beacon 또는 current_beacon_element_id 중 하나는 필수입니다."
                }
            )


        # 2) 목적지
        target_element_id = None
        target_node_id = None
        target_floor = None

        if req.target:
            target_element_id = req.target.element_id
            target_floor = req.target.floor
            target_node_id = (
                f"room_{target_element_id}_fl{target_floor}"
                if target_floor is not None else None
            )

        elif req.target_element_id:
            target_element_id = req.target_element_id

        elif req.target_node_id:
            target_node_id = req.target_node_id
            target_element_id = req.target_node_id

        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "target, target_element_id, target_node_id 중 하나는 필수입니다."
                }
            )


        # 3) 재난구역
        disaster_element_refs = []

        if req.disaster_elements:
            disaster_element_refs = req.disaster_elements

        elif req.disaster_element_ids:
            # 구버전 호환
            disaster_element_refs = req.disaster_element_ids


        # 4) 경로 계산
        result = compute_navigation_route(
            elements_json=elements_json,
            tags_map=req.tags_map or {},
            current_beacon_element_id=current_beacon_element_id,
            current_beacon_floor=current_beacon_floor,
            target_element_id=target_element_id,
            target_floor=target_floor,
            disaster_element_refs=disaster_element_refs,
            outline_bboxes=req.outline_bboxes,
            target_node_id=target_node_id,
        )
        # 🔍 디버그: target_node_id가 실제 그래프에 있었는지 + 같은 id를 가진 원본 element들의 type/floor 확인
        all_nodes_debug = result.get("_all_nodes", {})
        target_node_exists = (target_node_id in all_nodes_debug) if target_node_id else False

        matching_elements = [
            {"id": el.get("id"), "type": el.get("type"), "name": el.get("name"), "floor": el.get("floor")}
            for el in elements_json if el.get("id") == target_element_id
        ]
        result.pop("_all_nodes", None)
        result.pop("_path_ids", None)

        # 🔍 디버그: 응답에도 어떤 값이 쓰였는지 그대로 남김 (원인 파악되면 이 블록 지우면 됨)
        result["_debug"] = {
            "received_current_beacon": req.current_beacon.dict() if req.current_beacon else None,
            "received_target": req.target.dict() if req.target else None,
            "received_target_element_id": req.target_element_id,
            "received_target_node_id": req.target_node_id,
            "resolved_target_element_id": target_element_id,
            "resolved_target_node_id": target_node_id,
            "target_node_exists_in_graph": target_node_exists,
            "matching_elements_for_target_id": matching_elements,
        }
        return result
    except KeyError as e:
        return JSONResponse(status_code=400, content={"error": f"필수 파라미터 누락: {e}"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


# =============================================================
# [네비게이션] 백엔드 API로 실제 지도 데이터 조회 (JSON 직접 전달 대신)
#
# 확인된 사항:
# - 실제 시나리오에 쓰이는 지도는 학교 원본(channel_maps)이 아니라
#   해당 반(classroom)이 커스터마이징한 버전 -> GET /api/rooms/{classroomId}/map
# - 응답은 snake_case: elements_json(리스트), tags_map({element_id: {zone_type: ...}})
# =============================================================

MAP_API_PATH_TEMPLATE = os.environ.get(
    "MAP_API_PATH_TEMPLATE",
    "/api/rooms/{classroom_id}/map"
)


def fetch_classroom_map(classroom_id: str) -> "tuple[list, dict]":
    """반(classroom)의 현재 지도 데이터를 조회해서 (elements_json, tags_map) 튜플로 반환.
    학교 원본(channel_maps)이 아니라 이 반이 커스터마이징한 버전을 써야 실제 시나리오와 일치함."""
    url = f"{BACKEND_API_BASE}{MAP_API_PATH_TEMPLATE.format(classroom_id=classroom_id)}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    elements_json = data.get("elements_json", [])
    tags_map = data.get("tags_map", {})
    return elements_json, tags_map


class NavigationRouteAPIRequest(BaseModel):
    classroom_id: str

    # 신규 방식
    current_beacon: Optional[ElementRef] = None
    target: Optional[ElementRef] = None
    disaster_elements: Optional[List[ElementRef]] = None

    # 구버전 호환. ID가 여러 층에 중복되면 compute_navigation_route에서 400 처리한다.
    current_beacon_element_id: Optional[str] = None
    target_element_id: Optional[str] = None
    disaster_element_ids: Optional[List[str]] = None


@app.post("/route/from-api")
def navigation_route_from_api(req: NavigationRouteAPIRequest):
    """classroom_id로 백엔드 지도를 조회한 뒤 층+element_id 기준으로 경로를 계산."""
    try:
        elements_json, tags_map = fetch_classroom_map(req.classroom_id)
    except Exception as e:
        return JSONResponse(status_code=502, content={"detail": f"지도 API 조회 실패: {e}"})

    # 출발 위치 해석
    if req.current_beacon:
        current_beacon_element_id = req.current_beacon.element_id
        current_beacon_floor = req.current_beacon.floor
    elif req.current_beacon_element_id:
        current_beacon_element_id = req.current_beacon_element_id
        current_beacon_floor = None
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "current_beacon 또는 current_beacon_element_id 중 하나는 필수입니다."},
        )

    # 목적지 해석
    if req.target:
        target_element_id = req.target.element_id
        target_floor = req.target.floor
        target_node_id = (
            f"room_{target_element_id}_fl{target_floor}"
            if target_floor is not None else None
        )
    elif req.target_element_id:
        target_element_id = req.target_element_id
        target_floor = None
        target_node_id = None
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "target 또는 target_element_id 중 하나는 필수입니다."},
        )

    disaster_refs = list(req.disaster_elements or [])
    disaster_ids = list(req.disaster_element_ids or [])

    try:
        result = compute_navigation_route(
            elements_json=elements_json,
            tags_map=tags_map,
            current_beacon_element_id=current_beacon_element_id,
            current_beacon_floor=current_beacon_floor,
            target_element_id=target_element_id,
            target_floor=target_floor,
            target_node_id=target_node_id,
            disaster_element_refs=disaster_refs,
            disaster_element_ids=disaster_ids,
        )
        result.pop("_all_nodes", None)
        result.pop("_path_ids", None)
        return result
    except (KeyError, ValueError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


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
