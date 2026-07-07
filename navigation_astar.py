"""
navigation_astar.py — 재난훈련 A* 네비게이션 API
==================================================
test.py의 검증된 알고리즘을 그대로 이식. 시각화 코드만 제거.

[적용 조건]
  1. 방 투과 금지      : WP는 방 외부에만 생성, seg_blocked로 간선 차단
  2. 문 통과 필수      : 방↔문(같은 room_id)만 연결
  3. 계단 층간 연결    : 클릭으로 찍은 stair_positions를 모든 층에 등록, 인접 층 연결
  4. 안전구역 목적지   : restricted 가중치 대폭 상향
  5. 비콘 기반 출발지  : nearest_corridor로 가장 가까운 복도 노드를 출발점으로

[백엔드 API]
  python navigation_astar.py  →  Flask on :5001
  POST /route
    body: {
      elements_json: [...],          # channel_maps.elements_json
      tags_map: {...},               # {element_id: {zone_type, passable}}
      current_beacon_element_id: "", # students.last_beacon_id 기반
      target_element_id: "",         # 목적지 element_id
      disaster_element_ids: [...],   # 재난 구역 (optional)
      stair_positions: [{x,y}, ...]  # 계단 좌표 목록 (optional)
    }
  GET /health

[visualize.py 연동]
  visualize.py가 이 파일을 직접 import해서 compute_navigation_route() 호출
"""

from __future__ import annotations
import heapq, math, json
from typing import Any

# ─────────────────────────────────────────────────────────
# 상수 — test.py와 동일
# ─────────────────────────────────────────────────────────
CORRIDOR_STEP : float = 35.0
WP_MAX_DIST   : float = CORRIDOR_STEP * 1.45
CONN_DIST     : float = 120.0
DOOR_MARGIN   : float = 10.0
STAIR_COST    : float = 150.0

ZONE_WEIGHTS = {"safe": 1.0, "normal": 1.1, "restricted": 8.0, "danger": 50.0}


# ─────────────────────────────────────────────────────────
# 핵심 함수 — test.py 그대로
# ─────────────────────────────────────────────────────────

def pt_in_room(x, y, rooms, margin=1):
    for r in rooms:
        if (r["x"] + margin < x < r["x"] + r["w"] - margin and
                r["y"] + margin < y < r["y"] + r["h"] - margin):
            return r
    return None


def seg_blocked(x1, y1, x2, y2, rooms, samples=16):
    for i in range(1, samples):
        t = i / samples
        if pt_in_room(x1 + t*(x2-x1), y1 + t*(y2-y1), rooms, margin=1):
            return True
    return False


def door_exit_point(dx, dy, rooms):
    """문 위치에서 복도 방향 좌표 + 소속 방 id"""
    best_r, bd = None, 9999.0
    BOUNDARY = 20.0
    for r in rooms:
        on_top    = abs(dy - r["y"])            < BOUNDARY and r["x"]-5 <= dx <= r["x"]+r["w"]+5
        on_bottom = abs(dy - (r["y"]+r["h"]))   < BOUNDARY and r["x"]-5 <= dx <= r["x"]+r["w"]+5
        on_left   = abs(dx - r["x"])            < BOUNDARY and r["y"]-5 <= dy <= r["y"]+r["h"]+5
        on_right  = abs(dx - (r["x"]+r["w"]))   < BOUNDARY and r["y"]-5 <= dy <= r["y"]+r["h"]+5
        if on_top or on_bottom or on_left or on_right:
            best_r = r; break
    if not best_r:
        for r in rooms:
            dist = math.hypot(dx-(r["x"]+r["w"]/2), dy-(r["y"]+r["h"]/2))
            if dist < bd: bd, best_r = dist, r
    if not best_r:
        return dx, dy, None
    r = best_r
    faces = {
        "top":    (abs(dy - r["y"]),            dx, r["y"] - DOOR_MARGIN),
        "bottom": (abs(dy - (r["y"]+r["h"])),   dx, r["y"] + r["h"] + DOOR_MARGIN),
        "left":   (abs(dx - r["x"]),            r["x"] - DOOR_MARGIN, dy),
        "right":  (abs(dx - (r["x"]+r["w"])),   r["x"] + r["w"] + DOOR_MARGIN, dy),
    }
    _, cx, cy = min(faces.values(), key=lambda v: v[0])
    return cx, cy, r["id"]


def build_floor_graph(fd: dict) -> tuple[dict, list]:
    """
    test.py build_floor_graph 그대로.
    fd: {rooms, doors, exits, beacons, bbox, floor_index}
    """
    rooms = fd["rooms"]
    doors = fd["doors"]
    exits = fd["exits"]
    fi    = fd["floor_index"]
    bb    = fd["bbox"]

    nodes: dict[str, dict] = {}

    # ── 복도 격자 WP ──
    x0, y0, x1, y1 = bb["x0"], bb["y0"], bb["x1"], bb["y1"]
    gx = x0 + CORRIDOR_STEP / 2
    while gx <= x1:
        gy = y0 + CORRIDOR_STEP / 2
        while gy <= y1:
            if not pt_in_room(gx, gy, rooms):
                nid = f"wp_{fi}_{int(gx)}_{int(gy)}"
                nodes[nid] = {"x": gx, "y": gy, "zone": "normal",
                               "ntype": "waypoint", "room_id": None, "floor": fi}
            gy += CORRIDOR_STEP
        gx += CORRIDOR_STEP

    # ── 문 복도 포인트 ──
    for door in doors:
        cx, cy, rid = door_exit_point(door["x"], door["y"], rooms)
        if pt_in_room(cx, cy, rooms):
            for off in [20, 35, 50]:
                for ddx, ddy in [(off,0),(-off,0),(0,off),(0,-off)]:
                    if not pt_in_room(door["x"]+ddx, door["y"]+ddy, rooms):
                        cx, cy = door["x"]+ddx, door["y"]+ddy; break
                else:
                    continue
                break
        # room_id에 floor 포함 — 같은 element_id가 다른 층에 있어도 구분
        rid_with_floor = f"{rid}_fl{fi}" if rid else None
        nid = f"door_{door['id']}"
        nodes[nid] = {"x": cx, "y": cy, "zone": "normal", "ntype": "door",
                       "room_id": rid_with_floor, "floor": fi,
                       "door_x": door["x"], "door_y": door["y"]}

    # ── 비상구 ──
    for ex in exits:
        cx, cy = ex["x"], ex["y"]
        if pt_in_room(cx, cy, rooms):
            for off in [30, 50, 80]:
                moved = False
                for ddx, ddy in [(off,0),(-off,0),(0,off),(0,-off)]:
                    if not pt_in_room(cx+ddx, cy+ddy, rooms):
                        cx, cy = cx+ddx, cy+ddy; moved = True; break
                if moved: break
        nid = f"exit_{ex['id']}"
        nodes[nid] = {"x": cx, "y": cy, "zone": "safe", "ntype": "exit",
                       "room_id": None, "floor": fi}

    # ── 방 중심 (목적지/출발지 전용) ──
    for r in rooms:
        # nid에 floor 포함 — 같은 element_id가 다른 층에 있어도 구분
        nid = f"room_{r['id']}_fl{fi}"
        nodes[nid] = {
            "x": r["x"] + r["w"]/2, "y": r["y"] + r["h"]/2,
            "zone": r["zone"], "ntype": "room",
            "room_id": f"{r['id']}_fl{fi}",
            "floor": fi, "room_name": r["name"],
        }

    # ── 간선 ──
    edges: list[tuple[str,str,float]] = []
    nl = list(nodes.items())

    for i, (aid, na) in enumerate(nl):
        for bid, nb in nl[i+1:]:
            ax, ay = na["x"], na["y"]
            bx, by = nb["x"], nb["y"]
            dist   = math.hypot(ax-bx, ay-by)
            ta, tb = na["ntype"], nb["ntype"]

            # 방 중심 ↔ 문(같은 방)만 연결 [조건 2]
            if ta == "room" or tb == "room":
                rn = na if ta == "room" else nb
                dn = nb if ta == "room" else na
                if dn["ntype"] == "door" and dn.get("room_id") == rn["room_id"]:
                    # 방 중심→문 세그먼트가 다른 방을 통과하는지 검사
                    # 단, 자신의 방은 제외하고 검사
                    own_id = rn["room_id"]
                    rooms_excl = [r for r in rooms if f"{r['id']}_fl{fi}" != own_id]
                    if not seg_blocked(rn["x"], rn["y"], dn["x"], dn["y"], rooms_excl):
                        edges.append((aid, bid, dist))
                continue

            # WP ↔ WP: 격자 이웃만 [조건 1]
            if ta == "waypoint" and tb == "waypoint":
                if dist > WP_MAX_DIST: continue
            elif ta == "waypoint" or tb == "waypoint":
                if dist > CONN_DIST: continue
            else:
                if dist > CONN_DIST: continue

            if seg_blocked(ax, ay, bx, by, rooms): continue

            cost = dist * max(ZONE_WEIGHTS.get(na["zone"], 1.1),
                              ZONE_WEIGHTS.get(nb["zone"], 1.1))
            edges.append((aid, bid, cost))

    return nodes, edges


def build_all_graphs(
    floor_data: list[dict],
    stair_positions: list[dict],
) -> tuple[dict, dict]:
    """
    test.py build_all_graphs 그대로.
    stair_positions: [{x, y}, ...] — 모든 층 공통 좌표
    """
    all_nodes: dict[str, dict] = {}
    all_edges: list[tuple[str,str,float]] = []
    stair_ids_by_floor: dict[int, list] = {fd["floor_index"]: [] for fd in floor_data}

    for fd in floor_data:
        fi    = fd["floor_index"]
        rooms = fd["rooms"]

        # 계단 노드 [조건 3]
        for si, sp in enumerate(stair_positions):
            sx, sy = sp["x"], sp["y"]
            snid = f"stair_{si}_fl{fi}"
            all_nodes[snid] = {"x": sx, "y": sy, "zone": "normal",
                                "ntype": "stair", "room_id": None, "floor": fi}
            stair_ids_by_floor[fi].append((si, snid))

        lnodes, ledges = build_floor_graph(fd)
        all_nodes.update(lnodes)
        all_edges.extend(ledges)

        # 계단 ↔ 주변 복도 WP 연결 (seg_blocked 없음 — 계단은 방 관통 허용)
        for si, snid in stair_ids_by_floor[fi]:
            sn = all_nodes[snid]
            best_wps = []
            for nid, nd in lnodes.items():
                if nd["ntype"] not in ("waypoint", "exit"): continue
                dist = math.hypot(sn["x"]-nd["x"], sn["y"]-nd["y"])
                if dist > CONN_DIST * 2: continue
                best_wps.append((dist, nid))
            best_wps.sort()
            for dist, nid in best_wps[:6]:
                all_edges.append((snid, nid, dist))
                all_edges.append((nid, snid, dist))

    # 계단 층간 연결 (인접 층끼리)
    floor_indices = [fd["floor_index"] for fd in floor_data]
    for si in range(len(stair_positions)):
        for k in range(len(floor_indices) - 1):
            fa, fb = floor_indices[k], floor_indices[k+1]
            a_id = f"stair_{si}_fl{fa}"
            b_id = f"stair_{si}_fl{fb}"
            if a_id in all_nodes and b_id in all_nodes:
                all_edges.append((a_id, b_id, STAIR_COST))
                all_edges.append((b_id, a_id, STAIR_COST))

    graph: dict[str, list] = {}
    for a, b, cost in all_edges:
        graph.setdefault(a, []).append((b, cost))
        graph.setdefault(b, []).append((a, cost))

    return all_nodes, graph


def astar(all_nodes: dict, graph: dict, start_id: str, goal_id: str) -> list[str] | None:
    """test.py astar 그대로"""
    if start_id not in all_nodes or goal_id not in all_nodes: return None
    if start_id == goal_id: return [start_id]
    goal = all_nodes[goal_id]

    def h(nid: str) -> float:
        n = all_nodes[nid]
        return (math.hypot(n["x"]-goal["x"], n["y"]-goal["y"])
                + abs(n["floor"]-goal["floor"]) * 50)

    g: dict[str,float]       = {start_id: 0.0}
    came: dict[str,str|None] = {start_id: None}
    pq   = [(h(start_id), start_id)]
    vis: set[str] = set()

    while pq:
        _, cur = heapq.heappop(pq)
        if cur in vis: continue
        vis.add(cur)
        if cur == goal_id:
            path: list[str] = []
            nd: str | None = goal_id
            while nd is not None:
                path.append(nd); nd = came.get(nd)
            path.reverse(); return path
        for nb, cost in graph.get(cur, []):
            if nb in vis: continue
            tg = g[cur] + cost
            if tg < g.get(nb, math.inf):
                g[nb] = tg; came[nb] = cur
                heapq.heappush(pq, (tg + h(nb), nb))
    return None


def nearest_corridor(x: float, y: float, floor_index: int, all_nodes: dict) -> str | None:
    """test.py nearest_corridor 그대로"""
    best, bd = None, math.inf
    for nid, n in all_nodes.items():
        if n.get("floor") != floor_index: continue
        if n["ntype"] == "room": continue
        d = math.hypot(n["x"]-x, n["y"]-y)
        if d < bd: bd, best = d, nid
    return best


# ─────────────────────────────────────────────────────────
# elements_json 파싱 (백엔드 DB → 알고리즘 입력 변환)
# ─────────────────────────────────────────────────────────

def parse_elements(
    elements_json: list[dict],
    tags_map: dict[str, dict],
    outline_bboxes: dict[int, dict] | None = None,
) -> list[dict]:
    """
    elements_json + tags_map → floor_data 목록
    floor_data[i] = {floor_index, rooms, doors, exits, beacons, bbox}

    rooms 형식: {id, name, x, y, w, h, zone, floor_index}
    doors 형식: {id, x, y}
    exits 형식: {id, x, y}
    beacons 형식: {id, x, y}
    """
    ZONE_MAP = {"방":"normal","안전 구역":"safe","재난 구역":"danger","제한 구역":"restricted"}
    by_floor: dict[int, dict] = {}

    def get_floor(fi: int) -> dict:
        if fi not in by_floor:
            by_floor[fi] = {"floor_index": fi,
                             "rooms": [], "doors": [], "exits": [], "beacons": []}
        return by_floor[fi]

    for el in elements_json:
        t   = el.get("type", "")
        eid = el["id"]
        fi  = el.get("floor", 0)
        fd  = get_floor(fi)
        tags = tags_map.get(eid, {})
        zone = tags.get("zone_type", ZONE_MAP.get(t, "normal"))

        if t in ("방", "안전 구역", "재난 구역", "제한 구역"):
            fd["rooms"].append({
                "id":   eid,
                "name": el.get("name", t),
                "x":    float(el.get("x", 0)),
                "y":    float(el.get("y", 0)),
                "w":    float(el.get("width", 0)),
                "h":    float(el.get("height", 0)),
                "zone": zone,
                "floor_index": fi,
            })
        elif t == "문":
            fd["doors"].append({
                "id": eid,
                "x":  float(el.get("x", 0)),
                "y":  float(el.get("y", 0)),
            })
        elif t == "비상구":
            fd["exits"].append({
                "id": eid,
                "x":  float(el.get("x", 0)),
                "y":  float(el.get("y", 0)),
            })
        elif t == "비콘":
            fd["beacons"].append({
                "id":   eid,
                "x":    float(el.get("x", 0)),
                "y":    float(el.get("y", 0)),
                "zone": zone,
            })

    # bbox 계산
    for fi, fd in by_floor.items():
        ob = (outline_bboxes or {}).get(fi)
        if ob:
            fd["bbox"] = ob
        elif fd["rooms"]:
            x0 = min(r["x"] for r in fd["rooms"])
            y0 = min(r["y"] for r in fd["rooms"])
            x1 = max(r["x"]+r["w"] for r in fd["rooms"])
            y1 = max(r["y"]+r["h"] for r in fd["rooms"])
            fd["bbox"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        else:
            fd["bbox"] = {"x0": 0, "y0": 0, "x1": 1000, "y1": 1000}

    return sorted(by_floor.values(), key=lambda fd: fd["floor_index"])


# ─────────────────────────────────────────────────────────
# 메인 인터페이스
# ─────────────────────────────────────────────────────────

def compute_navigation_route(
    elements_json: list[dict[str, Any]],
    tags_map: dict[str, dict[str, Any]],
    current_beacon_element_id: str,
    target_element_id: str,
    disaster_element_ids: list[str] | None = None,
    stair_positions: list[dict] | None = None,
    outline_bboxes: dict[int, dict] | None = None,
    target_node_id: str | None = None,  # floor 포함 nid (visualize.py에서 전달)
) -> dict[str, Any]:
    """
    백엔드 호출 메인 함수.

    Args:
        elements_json:              channel_maps.elements_json
        tags_map:                   {element_id: {zone_type, passable}}
        current_beacon_element_id:  비콘 element_id [조건 5]
        target_element_id:          목적지 element_id
        disaster_element_ids:       재난 구역 element_id (zone→danger 강제)
        stair_positions:            [{x,y}] 계단 좌표 목록 [조건 3]
                                    - visualize.py: 클릭으로 전달
                                    - 실제 백엔드: DB에서 name="계단"인 방 좌표 전달
        outline_bboxes:             {floor_index: {x0,y0,x1,y1}} 건물 윤곽

    Returns:
        {
          path: [{element_id, name, cx, cy, zone_type, floor, ntype}, ...],
          start_element_id, goal_element_id,
          total_cost, found, warning
        }
    """
    # 재난 구역 오버라이드
    eff_tags: dict[str, dict] = {k: dict(v) for k, v in tags_map.items()}
    for dis_id in (disaster_element_ids or []):
        eff_tags.setdefault(dis_id, {})["zone_type"] = "danger"
        eff_tags[dis_id].setdefault("passable", True)

    # 파싱
    floor_data = parse_elements(elements_json, eff_tags, outline_bboxes)

    # 그래프 구성
    all_nodes, graph = build_all_graphs(floor_data, stair_positions or [])

    # [조건 5] 출발 노드 — 비콘 위치에서 가장 가까운 복도 노드
    bc_el = next((el for el in elements_json
                  if el["id"] == current_beacon_element_id), None)
    if bc_el:
        fi = bc_el.get("floor", 0)
        start_id = nearest_corridor(
            float(bc_el.get("x", 0)), float(bc_el.get("y", 0)),
            fi, all_nodes)
    else:
        start_id = None

    if not start_id:
        return _no_path(str(current_beacon_element_id), target_element_id)

    # 목적지 노드 — target_node_id가 있으면 그대로 사용 (floor 정보 포함)
    # 없으면 elements_json에서 같은 id 중 safe/danger zone인 것을 찾아 floor 결정
    if target_node_id and target_node_id in all_nodes:
        goal_id = target_node_id
    else:
        # elements_json에서 target_element_id + zone 기반으로 floor 찾기
        target_matches = [el for el in elements_json if el["id"] == target_element_id]
        target_el = None
        for el in target_matches:
            zone = tags_map.get(el["id"], {}).get("zone_type", "normal")
            if zone in ("safe", "danger", "restricted"):
                target_el = el; break
        if not target_el and target_matches:
            target_el = target_matches[-1]  # 마지막 층(가장 높은 floor) 우선
        target_fi = target_el.get("floor", 0) if target_el else 0
        goal_id = f"room_{target_element_id}_fl{target_fi}"
        if goal_id not in all_nodes:
            goal_id = f"exit_{target_element_id}"
    if goal_id not in all_nodes:
        return _no_path(start_id, target_element_id)

    # A*
    path_ids = astar(all_nodes, graph, start_id, goal_id)
    if not path_ids:
        return _no_path(start_id, goal_id)

    # 직렬화
    path_detail: list[dict] = []
    total = 0.0
    for i, nid in enumerate(path_ids):
        n = all_nodes[nid]
        path_detail.append({
            "element_id": nid,
            "name":       n.get("room_name", n["ntype"]),
            "cx":         round(n["x"], 2),
            "cy":         round(n["y"], 2),
            "zone_type":  n["zone"],
            "floor":      n["floor"],
            "ntype":      n["ntype"],
        })
        if i > 0:
            p = all_nodes[path_ids[i-1]]
            total += math.hypot(n["x"]-p["x"], n["y"]-p["y"])

    zones = {all_nodes[nid]["zone"] for nid in path_ids}
    warning = ("danger_zone_in_path"     if "danger"     in zones else
               "restricted_zone_in_path" if "restricted" in zones else None)

    return {
        "path":               path_detail,
        "start_element_id":   start_id,
        "goal_element_id":    goal_id,
        "total_cost":         round(total, 2),
        "found":              True,
        "warning":            warning,
        # visualize.py 렌더링용 (all_nodes, path_ids는 API 응답에서 제외)
        "_all_nodes":         all_nodes,
        "_path_ids":          path_ids,
    }


def _no_path(start: str, goal: str) -> dict:
    return {"path": [], "start_element_id": start, "goal_element_id": goal,
            "total_cost": None, "found": False, "warning": "no_path_found",
            "_all_nodes": {}, "_path_ids": []}


# ─────────────────────────────────────────────────────────
# Flask API 서버 (백엔드용)
# ─────────────────────────────────────────────────────────

def _run_api_server(port: int = 5001) -> None:
    try:
        from flask import Flask, jsonify, request as freq
    except ImportError:
        import subprocess as _sp, sys as _sys
        _sp.check_call([_sys.executable, "-m", "pip", "install", "flask"],
                       stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        from flask import Flask, jsonify, request as freq

    api = Flask("navigation_astar_api")

    @api.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @api.route("/route", methods=["POST"])
    def route():
        body = freq.get_json(force=True)
        try:
            result = compute_navigation_route(
                elements_json             = body["elements_json"],
                tags_map                  = body["tags_map"],
                current_beacon_element_id = body["current_beacon_element_id"],
                target_element_id         = body["target_element_id"],
                disaster_element_ids      = body.get("disaster_element_ids"),
                stair_positions           = body.get("stair_positions", []),
                outline_bboxes            = body.get("outline_bboxes"),
            )
            # API 응답에서 내부용 필드 제거
            result.pop("_all_nodes", None)
            result.pop("_path_ids", None)
            return jsonify(result)
        except KeyError as e:
            return jsonify({"error": f"필수 파라미터 누락: {e}"}), 400
        except Exception as e:
            import traceback
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    print("=" * 50)
    print(f"  navigation_astar API  http://localhost:{port}")
    print(f"  POST /route  |  GET /health")
    print("=" * 50)
    api.run(host="0.0.0.0", port=port, debug=False)


# ─────────────────────────────────────────────────────────
# __main__ : 간단 테스트 후 API 서버 실행
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    _run_api_server(port=5001)
