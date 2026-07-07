import uuid
from datetime import datetime

from app.models import (
    ScenarioActionEvents,
    ScenarioTeamMember,
    QuizSubmission
)


# =====================================================
# 🔥 소화기 퀴즈 제출 + 팀 전체 완료 체크
# =====================================================

def submit_fire_extinguisher_quiz(
    student_id,
    scenario_id,
    assignment_id,
    selected_answer,
    is_correct
):

    # -------------------------
    # 1️⃣ quiz_submissions 저장
    # -------------------------

    existing = QuizSubmission.objects.filter(
        scenario_id=scenario_id,
        assignment_id=assignment_id,
        student_id=student_id
    ).first()

    if not existing:

        QuizSubmission.objects.create(
            id=str(uuid.uuid4()),
            scenario_id=scenario_id,
            assignment_id=assignment_id,
            student_id=student_id,
            selected_answer=selected_answer,
            is_correct=is_correct,
            status=(
                "CORRECT"
                if is_correct
                else "FAILED"
            ),
            submitted_at=datetime.now()
        )

    # -------------------------
    # 2️⃣ action_event 기록 (점수 계산용)
    # -------------------------

    ScenarioActionEvents.objects.create(
        id=str(uuid.uuid4()),
        scenario_id=scenario_id,
        student_id=student_id,
        action_type="QUIZ_SUBMIT",
        meta_json={
            "quiz_type": "FIRE_EXTINGUISHER",
            "is_correct": is_correct
        },
        created_at=datetime.now()
    )

    # -------------------------
    # 3️⃣ 내 팀 찾기
    # -------------------------

    team_member = ScenarioTeamMember.objects.filter(
        scenario_id=scenario_id,
        student_id=student_id
    ).first()

    if not team_member:

        return {
            "status": "ERROR",
            "message": "팀 정보 없음"
        }

    team_id = team_member.team_id

    # -------------------------
    # 4️⃣ 팀 전체 인원 수
    # -------------------------

    total_members = ScenarioTeamMember.objects.filter(
        scenario_id=scenario_id,
        team_id=team_id
    ).count()

    # -------------------------
    # 5️⃣ 완료 인원 수
    # -------------------------

    team_students = ScenarioTeamMember.objects.filter(
        scenario_id=scenario_id,
        team_id=team_id
    ).values_list(
        "student_id",
        flat=True
    )

    completed_members = QuizSubmission.objects.filter(
        scenario_id=scenario_id,
        assignment_id=assignment_id,
        student_id__in=team_students
    ).values("student_id").distinct().count()

    # -------------------------
    # 6️⃣ 전원 완료 여부 판단
    # -------------------------

    if completed_members >= total_members:

        return {
            "status": "START_DONUT",
            "team_id": team_id
        }

    else:

        return {
            "status": "WAITING",
            "completed": completed_members,
            "total": total_members
        }


# =====================================================
# 🔄 대기 화면에서 계속 확인용
# =====================================================

def check_fire_team_ready(
    student_id,
    scenario_id,
    assignment_id
):

    team_member = ScenarioTeamMember.objects.filter(
        scenario_id=scenario_id,
        student_id=student_id
    ).first()

    if not team_member:

        return {"ready": False}

    team_id = team_member.team_id

    total_members = ScenarioTeamMember.objects.filter(
        scenario_id=scenario_id,
        team_id=team_id
    ).count()

    team_students = ScenarioTeamMember.objects.filter(
        scenario_id=scenario_id,
        team_id=team_id
    ).values_list(
        "student_id",
        flat=True
    )

    completed_members = QuizSubmission.objects.filter(
        scenario_id=scenario_id,
        assignment_id=assignment_id,
        student_id__in=team_students
    ).values("student_id").distinct().count()

    if completed_members >= total_members:

        return {
            "ready": True
        }

    return {
        "ready": False,
        "completed": completed_members,
        "total": total_members
    }


# =====================================================
# 🎯 기존 점수 계산 (네 코드 유지)
# =====================================================

def calculate_fire_role_score(
    student_id,
    scenario_id
):

    score = 0

    actions = ScenarioActionEvents.objects.filter(
        scenario_id=scenario_id,
        student_id=student_id
    )

    # -------------------------
    # 1️⃣ 소화기 획득 (10점)
    # -------------------------

    if actions.filter(
        action_type="PICKUP_ITEM",
        element_id="FIRE_EXTINGUISHER"
    ).exists():

        score += 10


    # -------------------------
    # 2️⃣ 소화기 퀴즈 성공 (20점)
    # -------------------------

    if actions.filter(
        action_type="QUIZ_SUBMIT",
        meta_json__quiz_type="FIRE_EXTINGUISHER",
        meta_json__is_correct=True
    ).exists():

        score += 20


    # -------------------------
    # 3️⃣ 도넛 참여 (10점)
    # -------------------------

    if actions.filter(
        action_type="MISSION_INCREMENT",
        element_id="FIRE_DONUT"
    ).exists():

        score += 10


    return score