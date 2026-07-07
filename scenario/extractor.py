from disaster.models import ChannelElementTag
from django.db.models.expressions import RawSQL


def _get_disaster_zone_queryset(school_id):
    """
    내부 공통 QuerySet 생성 함수
    JSON에서 zoneType 추출
    """

    return ChannelElementTag.objects.annotate(

        zone_type=RawSQL(
            """
            JSON_UNQUOTE(
                JSON_EXTRACT(
                    COALESCE(tags_json, '{}'),
                    '$.zoneType'
                )
            )
            """,
            []
        )

    ).filter(

        school_id=school_id,
        zone_type="DISASTER_ZONE"

    )


def has_disaster_zone(school_id):
    """
    재난구역 존재 여부 확인
    """

    return _get_disaster_zone_queryset(
        school_id
    ).exists()


def extract_disaster_zones(school_id):
    """
    재난구역 목록 반환
    """

    zones = _get_disaster_zone_queryset(
        school_id
    )

    results = []

    for z in zones:

        # name 없는 경우 방어
        if not z.name:
            continue

        results.append({

            "room_name": z.name,

            "floor": z.floor_index

        })

    return results