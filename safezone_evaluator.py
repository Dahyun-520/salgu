from .models import StudentMissionProgress


# =====================================================
# 1️⃣ SAFE_ZONE 진입 여부 확인 (True / False)
# =====================================================

def check_safe_zone_success(
    student_id,
    scenario_id
):

    return StudentMissionProgress.objects.filter(
        student_id=student_id,
        scenario_id=scenario_id,
        status="COMPLETED",
        assignment__params_json__missionCode="COMMON_SAFE_ZONE"
    ).exists()


# =====================================================
# 2️⃣ SAFE_ZONE 점수 계산 (0 or 10)
# =====================================================

def calculate_safezone_score(
    student_id,
    scenario_id
):

    entered = check_safe_zone_success(
        student_id,
        scenario_id
    )

    if entered:

        return 10

    return 0