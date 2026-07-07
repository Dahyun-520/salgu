"""
재난구역 추출 전용 모듈

역할:
- 구조도 JSON에서 재난구역만 선별
- 시나리오 / 미션 / 시뮬레이션 공통 사용
"""

from typing import List, Dict


def extract_disaster_zones(structure_json: Dict) -> List[Dict]:
    """
    구조도 JSON에서 재난구역 요소만 추출한다.

    Returns:
        [
            {
                "element_id": "...",
                "room_name": "...",
                "floor": 2,
                "x": 67,
                "y": 78,
                "width": 150,
                "height": 46
            }
        ]
    """

    zones = []

    elements = structure_json.get("elements", [])
    for el in elements:
        # 1️⃣ 재난 구역만
        if el.get("type") != "재난 구역":
            continue

        # 2️⃣ 이름 없는 건 스킵 (AI 분류 불가)
        room_name = el.get("name")
        if not room_name:
            continue

        zone = {
            "element_id": el.get("id"),
            "room_name": room_name,
            "floor": el.get("floor"),

            # 위치 정보 (미션, UI 하이라이트용)
            "x": el.get("x"),
            "y": el.get("y"),
            "width": el.get("width"),
            "height": el.get("height"),
        }

        zones.append(zone)

    return zones


def has_disaster_zone(structure_json: Dict) -> bool:
    """
    재난구역이 하나라도 존재하는지 확인
    (시나리오 관리 페이지 진입 전 체크용)
    """
    return any(
        el.get("type") == "재난 구역"
        for el in structure_json.get("elements", [])
    )