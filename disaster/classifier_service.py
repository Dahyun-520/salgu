from disaster.repository import find_zone, save_zone
from scenario.ai_classifier import classify_place


def get_or_create_classification(room_name: str, floor: int) -> str:
    """
    재난구역 장소 분류 서비스
    - 이미 분류된 경우: DB(임시)에서 반환
    - 없으면: AI 분류 → 저장 → 반환
    """

    existing = find_zone(room_name, floor)
    if existing:
        return existing["classified_place"]

    classified_place = classify_place(room_name)
    save_zone(room_name, floor, classified_place)

    return classified_place