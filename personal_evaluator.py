from django.db.models import Exists, OuterRef

from .models import (
    QuizSubmission,
    StudentMissionProgress,
    StudentItem,
    Item,
    ScenarioAssignment,
    Content
)


def calculate_personal_score(student_id, scenario_id):

    score = 0

    # -------------------------
    # 1️⃣ 랜덤 퀴즈 성공 (8점)
    # -------------------------

    quiz_success = QuizSubmission.objects.filter(
        student_id=student_id,
        scenario_id=scenario_id,
        status="CORRECT"
    ).exists()

    if quiz_success:
        score += 8

    # -------------------------
    # 2️⃣ 전화 미션 완료 (6점)
    # -------------------------

    phone_mission_success = StudentMissionProgress.objects.filter(
        student_id=student_id,
        scenario_id=scenario_id,
        status="COMPLETED",
        assignment__content__title="전화 미션"
    ).exists()

    if phone_mission_success:
        score += 6

    # -------------------------
    # 3️⃣ 소화기 획득 (6점)
    # -------------------------

    extinguisher_success = StudentItem.objects.filter(
        student_id=student_id,
        scenario_id=scenario_id,
        is_consumed=False,
        item__item_code="EXTINGUISHER"
    ).exists()

    if extinguisher_success:
        score += 6

    return score