from app.database import get_db
from app.models import Contents


def get_quiz_by_id(content_id):

    db = next(get_db())

    quiz = (
        db.query(Contents)
        .filter(
            Contents.id == content_id,
            Contents.content_type == "QUIZ"
        )
        .first()
    )

    if not quiz:
        return None

    return {
        "quiz_id": quiz.id,
        "type": quiz.quiz_type,   # MCQ / OX
        "question": quiz.question,
        "choices": quiz.choices_json,
        "answer": quiz.answer
    }