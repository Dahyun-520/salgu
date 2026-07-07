from disaster.classifier_service import (
    get_or_create_classification
)

from scenario.scenario_repository import (
    get_random_scenario_from_db
)


def has_disaster_zone(school_id):

    # 테스트용 강제 True
    return True


def extract_disaster_zones(school_id):

    # 테스트용 하드코딩 재난구역
    return [

        {
            "room_name": "컴퓨터실",
            "floor": 3
        },

        {
            "room_name": "학생 휴게 공간",
            "floor": 1
        },

        {
            "room_name": "과학 융합 실습실",
            "floor": 2
        },

        {
            "room_name": "3학년 1반",
            "floor": 1
        }

    ]


def generate_scenarios_from_structure(

    school_id: str,

    scenario_setting_json: dict,

    scenario_instance=None

):

    if not has_disaster_zone(school_id):

        return {

            "status": "WAIT",

            "message": "재난구역이 지정되지 않았습니다."

        }

    scenario_type = scenario_setting_json["scenarioType"]

    results = []

    zones = extract_disaster_zones(school_id)

    ai_logs = []

    for zone in zones:

        room_name = zone["room_name"]

        floor = zone["floor"]

        print()
        print("===== 장소 분류 시작 =====")

        print("원본 장소명:", room_name)

        # 1️⃣ 장소 분류
        place_type = get_or_create_classification(

            room_name,
            floor

        )

        print("AI 분류 결과:", place_type)

        # 2️⃣ CSV 기반 시나리오 선택
        content = get_random_scenario_from_db(

            place_type,
            scenario_type

        )

        if not content:

            print("시나리오 없음")
            continue

        print("선택된 시나리오:", content.reason)

        ai_logs.append({

            "original": room_name,

            "classified": place_type

        })

        results.append({

            "floor": floor,

            "roomName": room_name,

            "classifiedPlace": place_type,

            "scenario": content.reason,

            "contentId": content.id

        })

        # scenario_instance 없을 때 방어
        if scenario_instance:

            scenario_instance.selected_scenario_event_content_id = content.id

    if scenario_instance:

        scenario_instance.ai_decision_json = ai_logs
        scenario_instance.save()

    print()
    print("===== 최종 결과 =====")

    for r in results:

        print(r)

    return {

        "status": "OK",

        "data": results

    }


# 단독 실행 테스트
if __name__ == "__main__":

    result = generate_scenarios_from_structure(

        school_id=1,

        scenario_setting_json={
            "scenarioType": "FIRE"
        }

    )

    print()
    print(result)