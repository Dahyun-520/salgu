import csv
import uuid
from contents.models import Content


def import_csv_to_db(
    csv_path,
    scenario_type
):

    with open(csv_path, encoding="cp949") as f:

        reader = csv.DictReader(f)

        for row in reader:

            Content.objects.create(

                id=str(uuid.uuid4()),

                content_type="SCENARIO_EVENT",

                scenario_type=scenario_type,

                place=row["장소"],

                reason=row["사유"]

            )