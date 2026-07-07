from .quiz_loader import get_db_connection


def calculate_quiz_score(
    student_id,
    scenario_id
):

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    query = """
        SELECT
            qs.assignment_id

        FROM quiz_submissions qs

        JOIN scenario_assignments sa
            ON qs.assignment_id = sa.id

        WHERE
            qs.student_id = %s
            AND sa.scenario_id = %s
            AND sa.assignment_type = 'QUIZ'
            AND qs.is_correct = 1
    """

    cursor.execute(
        query,
        (student_id, scenario_id)
    )

    rows = cursor.fetchall()

    correct_count = len(rows)

    score = correct_count * 6

    cursor.close()
    db.close()

    return score