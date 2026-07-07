"""
재난구역 분류 결과 저장소 (임시)

나중에 DB로 교체 예정
"""

_disaster_zone_table = []


def find_zone(room_name: str, floor: int):
    for row in _disaster_zone_table:
        if row["room_name"] == room_name and row["floor"] == floor:
            return row
    return None


def save_zone(room_name: str, floor: int, classified_place: str):
    row = {
        "room_name": room_name,
        "floor": floor,
        "classified_place": classified_place
    }
    _disaster_zone_table.append(row)
    return row
