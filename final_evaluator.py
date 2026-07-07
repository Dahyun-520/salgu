from .quiz_evaluator import calculate_quiz_score
from .role_evaluator import calculate_role_score
from .personal_evaluator import calculate_personal_score
from .safezone_evaluator import calculate_safezone_score


def calculate_total_score(student_id, scenario_id):

    quiz_score = calculate_quiz_score(
        student_id,
        scenario_id
    )

    role_score = calculate_role_score(
        student_id,
        scenario_id
    )

    personal_score = calculate_personal_score(
        student_id,
        scenario_id
    )

    safe_score = calculate_safezone_score(
        student_id,
        scenario_id
    )

    total = (
        quiz_score +
        role_score +
        personal_score +
        safe_score
    )

    return {
        "quiz": quiz_score,
        "role": role_score,
        "personal": personal_score,
        "safezone": safe_score,
        "total": total
    }