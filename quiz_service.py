import random
from datetime import datetime

from app.database import get_db
from app.models import (
    Scenario,
    ScenarioAssignment,
    QuizSubmission,
)

from app.quiz_loader import get_quiz_by_id


MAX_RANDOM_QUIZ = 5


class QuizService:

    @staticmethod
    def get_random_quiz_for_student(
        scenario_id: str,
        student_id: str,
    ):

        db = next(get_db())

        # -----------------------------
        # 시나리오 조회
        # -----------------------------

        scenario = (
            db.query(Scenario)
            .filter(
                Scenario.id == scenario_id
            )
            .first()
        )

        if not scenario:
            return None

        if not scenario.training_started_at:
            return None

        # -----------------------------
        # 시간 계산
        # -----------------------------

        train_time = scenario.train_time * 60

        interval = train_time // 6

        if interval <= 0:
            return None

        now = datetime.now()

        elapsed_time = (
            now - scenario.training_started_at
        ).total_seconds()

        available_count = int(
            elapsed_time // interval
        )

        if available_count <= 0:
            return None

        if available_count > MAX_RANDOM_QUIZ:
            available_count = MAX_RANDOM_QUIZ

        # -----------------------------
        # 이미 제출한 것 조회
        # -----------------------------

        solved = (
            db.query(
                QuizSubmission.assignment_id
            )
            .filter(
                QuizSubmission.student_id
                == student_id
            )
            .all()
        )

        solved_ids = [
            s.assignment_id
            for s in solved
        ]

        # -----------------------------
        # 퀴즈 assignment 조회
        # -----------------------------

        assignments = (
            db.query(ScenarioAssignment)
            .filter(
                ScenarioAssignment.scenario_id
                == scenario_id,
                ScenarioAssignment.assignment_type
                == "QUIZ",
            )
            .all()
        )

        assignments = [
            a
            for a in assignments
            if a.id not in solved_ids
        ]

        assignments = assignments[
            :available_count
        ]

        if not assignments:
            return None

        # -----------------------------
        # 랜덤 선택
        # -----------------------------

        assignment = random.choice(
            assignments
        )

        # -----------------------------
        # DB에서 퀴즈 가져오기
        # -----------------------------

        quiz = get_quiz_by_id(
            assignment.content_id
        )

        if not quiz:
            return None

        # -----------------------------
        # 목숨 설정
        # -----------------------------

        lives = (
            3
            if quiz["type"] == "MCQ"
            else 1
        )

        return {
            "assignment_id": assignment.id,
            "quiz_id": quiz["quiz_id"],
            "type": quiz["type"],
            "question": quiz["question"],
            "choices": quiz["choices"],
            "lives": lives,
        }