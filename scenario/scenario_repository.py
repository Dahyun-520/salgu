import random
import csv
import os


BASE_DIR = os.path.dirname(__file__)

CSV_PATH = os.path.join(
    BASE_DIR,
    "scenarios",
    "fire.csv"
)


class DummyContent:

    def __init__(self, reason):

        self.reason = reason
        self.id = "TEST_CONTENT_ID"


def get_random_scenario_from_db(

    place_type,
    scenario_type

):

    candidates = []

    with open(CSV_PATH, encoding="cp949") as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row["장소"] == place_type:

                candidates.append(

                    DummyContent(
                        row["사유"]
                    )

                )

    if not candidates:

        return None

    return random.choice(candidates)