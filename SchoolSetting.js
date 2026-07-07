// src/pages/SchoolSetting.js
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Navbar from "../components/Navbar";

export default function SchoolSetting() {
  // ====== Viewport ======
  const VIEW_W = 900;
  const VIEW_H = 600;
  const MIN_ZOOM = 0.5;

  const fileInputRef = useRef(null);
  const viewportRef = useRef(null);
  const menuRef = useRef(null);
  const [autoAnalyzing, setAutoAnalyzing] = useState(false);

  // ====== Helpers ======
  const clamp = useCallback((v, a, b) => Math.max(a, Math.min(b, v)), []);
  const round4 = (n) =>
    typeof n === "number" ? Math.round(n * 10000) / 10000 : n;

  // 색상 헬퍼
  function typeColor(type, alpha = 0.25) {
    switch (type) {
      case "제한 구역":
        return `rgba(255,165,0,${alpha})`;
      case "재난 구역":
        return `rgba(255,0,0,${alpha})`;
      case "안전 구역":
        return `rgba(0,200,0,${alpha})`;
      case "방":
        return `rgba(0,112,255,${alpha})`;
      case "복도":
        return `rgba(160,160,160,${alpha})`;
      default:
        return `rgba(0,0,0,0)`;
    }
  }

  // ====== Floor ======
  const makeEmptyFloor = useCallback(() => {
    return {
      imageSrc: null,
      uploadedFile: null, // 프론트 전용이지만 일단 유지
      imgNatural: { w: 0, h: 0 },
      elements: [],
      undoStack: [],
      redoStack: [],
      // (자동분석 캐시 같은 건 프론트 전용이라도 UI용으로 유지 가능)
      hasAutoAnalysisResult: false,
      autoElementsCache: [],
    };
  }, []);

  const [currentFloorIndex, setCurrentFloorIndex] = useState(0);
  const [floorNames, setFloorNames] = useState(["1층"]);
  const [floors, setFloors] = useState([makeEmptyFloor()]);

  const currentFloor = floors[currentFloorIndex] || makeEmptyFloor();
  const imageSrc = currentFloor.imageSrc;
  const imgNatural = currentFloor.imgNatural || { w: 0, h: 0 };

  // ✅ elements는 useMemo로 고정 (eslint warning 방지)
  const elements = useMemo(() => {
    return currentFloor.elements ?? [];
  }, [currentFloor.elements]);

  // 현재 층만 업데이트하는 setter
  const setElements = useCallback(
    (updater) => {
      setFloors((prev) => {
        const next = [...prev];
        const curr = next[currentFloorIndex] || makeEmptyFloor();
        const prevEls = curr.elements || [];
        const nextEls =
          typeof updater === "function" ? updater(prevEls) : updater;
        next[currentFloorIndex] = { ...curr, elements: nextEls };
        return next;
      });
    },
    [currentFloorIndex, makeEmptyFloor]
  );

  const pushUndoSnapshot = useCallback(() => {
    setFloors((prev) => {
      const next = [...prev];
      const curr = next[currentFloorIndex] || makeEmptyFloor();
      const snap = JSON.parse(JSON.stringify(curr.elements || []));
      next[currentFloorIndex] = {
        ...curr,
        undoStack: [...(curr.undoStack || []), snap],
        redoStack: [],
      };
      return next;
    });
  }, [currentFloorIndex, makeEmptyFloor]);

  const performUndoOnce = useCallback(() => {
    setFloors((prev) => {
      const next = [...prev];
      const curr = next[currentFloorIndex] || makeEmptyFloor();
      const undo = curr.undoStack || [];
      if (!undo.length) return prev;

      const last = undo[undo.length - 1];
      const rest = undo.slice(0, -1);

      next[currentFloorIndex] = {
        ...curr,
        elements: last,
        undoStack: rest,
        redoStack: [
          ...(curr.redoStack || []),
          JSON.parse(JSON.stringify(curr.elements || [])),
        ],
      };
      return next;
    });
  }, [currentFloorIndex, makeEmptyFloor]);
  const handleAutoAnalyze = useCallback(async () => {
    if (!imageSrc) {
      alert("먼저 이미지를 업로드해 주세요.");
      return;
    }

    // 현재 층의 업로드된 파일 가져오기
    const currentFile = floors[currentFloorIndex]?.uploadedFile;
    if (!currentFile) {
      alert("업로드된 이미지 파일을 찾을 수 없습니다. 이미지를 다시 업로드해 주세요.");
      return;
    }

    setAutoAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append("image", currentFile);

      const response = await fetch("https://getaway-deem-drop-down.ngrok-free.dev/analyze-floorplan", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `서버 오류: ${response.status}`);
      }

      const data = await response.json();
      const serverElements = data.elements || [];

      if (!serverElements.length) {
        alert("분석 결과가 없습니다. 이미지를 확인해 주세요.");
        return;
      }

      pushUndoSnapshot();

      // 서버 결과를 현재 층 elements로 세팅 (기존 요소는 유지하고 자동분석 결과 추가)
      setElements((prev) => {
        // 이전 자동분석 결과(source: SAM / OpenCV) 제거 후 새로 추가
        const manual = prev.filter(
          (el) => el.source !== "SAM" && el.source !== "OpenCV" && el.id !== "auto-outline"
        );
        const withFloor = serverElements.map((el) => ({
          ...el,
          floor: currentFloorIndex,
        }));
        return [...manual, ...withFloor];
      });

      alert(`✅ 자동 분석 완료! 요소 ${serverElements.length}개가 추가되었습니다.\nOCR: ${data.ocr_available ? "활성" : "비활성"} / 장치: ${data.sam_device}`);

    } catch (err) {
      console.error("[자동분석 오류]", err);
      alert(`❌ 자동 분석 실패: ${err.message}\n\n서버가 실행 중인지 확인해 주세요.\n(python server.py)`);
    } finally {
      setAutoAnalyzing(false);
    }
  }, [imageSrc, floors, currentFloorIndex, pushUndoSnapshot, setElements]);

  const performRedoOnce = useCallback(() => {
    setFloors((prev) => {
      const next = [...prev];
      const curr = next[currentFloorIndex] || makeEmptyFloor();
      const redos = curr.redoStack || [];
      if (!redos.length) return prev;

      const last = redos[redos.length - 1];
      const rest = redos.slice(0, -1);

      next[currentFloorIndex] = {
        ...curr,
        elements: last,
        redoStack: rest,
        undoStack: [
          ...(curr.undoStack || []),
          JSON.parse(JSON.stringify(curr.elements || [])),
        ],
      };
      return next;
    });
  }, [currentFloorIndex, makeEmptyFloor]);

  // ====== UI State ======
  const modeButtons = ["제한 구역", "방", "문", "비상구", "건물윤곽"];
  const [mode, setMode] = useState(null); // null / 위 버튼들
  const modeRef = useRef(mode);
  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  const [showImage, setShowImage] = useState(true);
  const [imageLoaded, setImageLoaded] = useState(false);

  // 선택
  const [selectedId, setSelectedId] = useState(null);
  const [selectedIds, setSelectedIds] = useState([]);
  const selectedIdRef = useRef(selectedId);
  const selectedIdsRef = useRef(selectedIds);
  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);
  useEffect(() => {
    selectedIdsRef.current = selectedIds;
  }, [selectedIds]);

  // 도움말
  const [showHelp, setShowHelp] = useState(false);

  // 컨텍스트 메뉴
  const [contextMenu, setContextMenu] = useState(null);

  // ====== Zoom & Pan ======
  const [fitScale, setFitScale] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [imgOffset, setImgOffset] = useState({ x: 0, y: 0 });
  const imgOffsetRef = useRef(imgOffset);
  useEffect(() => {
    imgOffsetRef.current = imgOffset;
  }, [imgOffset]);

  const displayedScale = fitScale * zoom;
  const displayedW = (imgNatural.w || 0) * displayedScale;
  const displayedH = (imgNatural.h || 0) * displayedScale;
  const baseLeft = (VIEW_W - displayedW) / 2;
  const baseTop = (VIEW_H - displayedH) / 2;
  const imgLeft = baseLeft + imgOffset.x;
  const imgTop = baseTop + imgOffset.y;

  const clampOffsetForScale = useCallback(
    (ox, oy, newScale) => {
      const w = (imgNatural.w || 0) * newScale;
      const h = (imgNatural.h || 0) * newScale;
      const newBaseLeft = (VIEW_W - w) / 2;
      const newBaseTop = (VIEW_H - h) / 2;

      // X
      if (w <= VIEW_W) {
        ox = 0;
      } else {
        const minLeft = VIEW_W - w;
        const maxLeft = 0;
        const nextLeft = newBaseLeft + ox;
        if (nextLeft > maxLeft) ox = maxLeft - newBaseLeft;
        else if (nextLeft < minLeft) ox = minLeft - newBaseLeft;
      }

      // Y
      if (h <= VIEW_H) {
        oy = 0;
      } else {
        const minTop = VIEW_H - h;
        const maxTop = 0;
        const nextTop = newBaseTop + oy;
        if (nextTop > maxTop) oy = maxTop - newBaseTop;
        else if (nextTop < minTop) oy = minTop - newBaseTop;
      }

      return { x: ox, y: oy };
    },
    [imgNatural.h, imgNatural.w]
  );

  const applyZoomAroundPoint = useCallback(
    (oldZoom, newZoom, centerClient) => {
      if (!viewportRef.current || !centerClient || !imageSrc || !imageLoaded)
        return;

      const rect = viewportRef.current.getBoundingClientRect();
      const clientX = centerClient.x;
      const clientY = centerClient.y;

      const oldScale = fitScale * oldZoom;
      const newScale = fitScale * newZoom;

      const oldW = (imgNatural.w || 0) * oldScale;
      const oldH = (imgNatural.h || 0) * oldScale;
      const oldBaseLeft = (VIEW_W - oldW) / 2;
      const oldBaseTop = (VIEW_H - oldH) / 2;

      const localX =
        clientX - rect.left - (oldBaseLeft + imgOffsetRef.current.x);
      const localY = clientY - rect.top - (oldBaseTop + imgOffsetRef.current.y);

      const natX = localX / oldScale;
      const natY = localY / oldScale;

      const newW = (imgNatural.w || 0) * newScale;
      const newH = (imgNatural.h || 0) * newScale;
      const newBaseLeft = (VIEW_W - newW) / 2;
      const newBaseTop = (VIEW_H - newH) / 2;

      const newLocalX = natX * newScale;
      const newLocalY = natY * newScale;

      const desiredOffsetX = clientX - rect.left - newBaseLeft - newLocalX;
      const desiredOffsetY = clientY - rect.top - newBaseTop - newLocalY;

      const clamped = clampOffsetForScale(
        desiredOffsetX,
        desiredOffsetY,
        newScale
      );
      setImgOffset(clamped);
    },
    [
      clampOffsetForScale,
      fitScale,
      imageLoaded,
      imageSrc,
      imgNatural.h,
      imgNatural.w,
    ]
  );

  const zoomIn = useCallback(() => {
    if (!viewportRef.current) return;
    const r = viewportRef.current.getBoundingClientRect();
    const center = { x: r.left + VIEW_W / 2, y: r.top + VIEW_H / 2 };

    setZoom((z) => {
      const next = clamp(z + 0.1, MIN_ZOOM, 10);
      if (next !== z) applyZoomAroundPoint(z, next, center);
      return next;
    });
  }, [applyZoomAroundPoint, clamp]);

  const zoomOut = useCallback(() => {
    if (!viewportRef.current) return;
    const r = viewportRef.current.getBoundingClientRect();
    const center = { x: r.left + VIEW_W / 2, y: r.top + VIEW_H / 2 };

    setZoom((z) => {
      const next = clamp(z - 0.1, MIN_ZOOM, 10);
      if (next !== z) applyZoomAroundPoint(z, next, center);
      return next;
    });
  }, [applyZoomAroundPoint, clamp]);

  // nat <-> disp
  const natToDispX = (natX) => imgLeft + natX * displayedScale;
  const natToDispY = (natY) => imgTop + natY * displayedScale;
  const natToDispW = (natW) => natW * displayedScale;
  const natToDispH = (natH) => natH * displayedScale;

  const clientToNatural = useCallback(
    (clientX, clientY) => {
      const rect = viewportRef.current.getBoundingClientRect();
      const localX = clientX - rect.left - imgLeft;
      const localY = clientY - rect.top - imgTop;
      return { x: localX / displayedScale, y: localY / displayedScale };
    },
    [displayedScale, imgLeft, imgTop]
  );

  const isInsideImageNatural = useCallback(
    (n) =>
      n.x >= 0 &&
      n.y >= 0 &&
      n.x <= (imgNatural.w || 0) &&
      n.y <= (imgNatural.h || 0),
    [imgNatural.h, imgNatural.w]
  );

  // Ctrl+wheel zoom (viewport 안에서만)
  useEffect(() => {
    const handler = (e) => {
      const rect = viewportRef.current?.getBoundingClientRect();
      if (!rect) return;
      if (!imageSrc || !imageLoaded) return;

      // 뷰포트 밖 무시
      if (
        e.clientX < rect.left ||
        e.clientX > rect.right ||
        e.clientY < rect.top ||
        e.clientY > rect.bottom
      )
        return;

      if (!e.ctrlKey) return;
      e.preventDefault();

      const oldZoom = zoom;
      const nextZoom = clamp(oldZoom - e.deltaY * 0.0015, MIN_ZOOM, 10);
      if (nextZoom === oldZoom) return;

      applyZoomAroundPoint(oldZoom, nextZoom, { x: e.clientX, y: e.clientY });
      setZoom(nextZoom);
    };

    window.addEventListener("wheel", handler, { passive: false });
    return () => window.removeEventListener("wheel", handler);
  }, [applyZoomAroundPoint, clamp, imageLoaded, imageSrc, zoom]);

  // 일반 휠 = 세로 스크롤(패닝)
  const onViewportWheel = useCallback(
    (e) => {
      if (!imageSrc || !imageLoaded) return;
      if (e.ctrlKey) return;

      // 세로로 넘칠 때만
      if (displayedH <= VIEW_H) return;

      e.preventDefault();
      const deltaY = e.deltaY;

      const newScale = fitScale * zoom;
      const desiredY = imgOffsetRef.current.y - deltaY;
      const clamped = clampOffsetForScale(
        imgOffsetRef.current.x,
        desiredY,
        newScale
      );
      setImgOffset(clamped);
    },
    [clampOffsetForScale, displayedH, fitScale, imageLoaded, imageSrc, zoom]
  );

  // ====== Space Panning ======
  const spacePressedRef = useRef(false);
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef(null);
  const panOriginRef = useRef(null);

  // ====== Drawing / Preview ======
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState(null);
  const [preview, setPreview] = useState(null);

  // ✅ (선택) drawStartRef로 안정적으로 쓰고 싶으면 아래도 가능
  // const drawStartRef = useRef(drawStart);
  // useEffect(() => { drawStartRef.current = drawStart; }, [drawStart]);

  // ====== Drag selection (box select) ======
  const [isBoxSelecting, setIsBoxSelecting] = useState(false);
  const [selectionRect, setSelectionRect] = useState(null);
  const selectionStartRef = useRef(null);

  // ====== Hover ======
  const [hover, setHover] = useState(null);

  // ====== Clipboard ======
  const [clipboard, setClipboard] = useState(null);
  const clipboardRef = useRef(clipboard);
  useEffect(() => {
    clipboardRef.current = clipboard;
  }, [clipboard]);

  // ====== Resize ======
  const [resizing, setResizing] = useState(null); // { id, handle, startNat, orig }
  const [editingResizeId, setEditingResizeId] = useState(null);

  // ====== Door rotate ======
  const [rotatingDoorId, setRotatingDoorId] = useState(null);
  const [pendingDoorId, setPendingDoorId] = useState(null);

  // ====== Outline drawing ======
  const [outlineRawPoints, setOutlineRawPoints] = useState([]);
  const [outlinePoints, setOutlinePoints] = useState([]);
  const outlineRawPointsRef = useRef(outlineRawPoints);
  useEffect(() => {
    outlineRawPointsRef.current = outlineRawPoints;
  }, [outlineRawPoints]);

  // ✅ snapAngleTo45를 useCallback으로 고정(= finalizeOutlineFromRaw 안정화)
  const snapAngleTo45 = useCallback((rad) => {
    const step = Math.PI / 4;
    return Math.round(rad / step) * step;
  }, []);

  // ✅ finalizeOutlineFromRaw도 useCallback으로 고정
  const finalizeOutlineFromRaw = useCallback(
    (rawPoints, closed = true) => {
      if (!rawPoints || rawPoints.length < 2)
        return { points: rawPoints || [], rawPoints: rawPoints || [] };

      const snapped = [];
      snapped.push({ ...rawPoints[0] });

      for (let i = 1; i < rawPoints.length; i++) {
        const prevRaw = rawPoints[i - 1];
        const currRaw = rawPoints[i];
        const prevSnap = snapped[i - 1];

        const dx = currRaw.x - prevRaw.x;
        const dy = currRaw.y - prevRaw.y;
        const len = Math.hypot(dx, dy);
        if (len === 0) {
          snapped.push({ ...prevSnap });
          continue;
        }
        const angle = Math.atan2(dy, dx);
        const snapA = snapAngleTo45(angle);
        snapped.push({
          x: prevSnap.x + len * Math.cos(snapA),
          y: prevSnap.y + len * Math.sin(snapA),
        });
      }

      if (closed && rawPoints.length >= 2) {
        const n = rawPoints.length;
        const rawLast = rawPoints[n - 1];
        const rawFirst = rawPoints[0];
        const dx = rawFirst.x - rawLast.x;
        const dy = rawFirst.y - rawLast.y;
        const len = Math.hypot(dx, dy);
        if (len > 0) {
          const angle = Math.atan2(dy, dx);
          const snapA = snapAngleTo45(angle);
          const newLastX = snapped[0].x - len * Math.cos(snapA);
          const newLastY = snapped[0].y - len * Math.sin(snapA);
          snapped[n - 1] = { x: newLastX, y: newLastY };
        }
      }

      return { points: snapped, rawPoints: [...rawPoints] };
    },
    [snapAngleTo45]
  );

  // ✅ finalizeCurrentOutline deps에 finalizeOutlineFromRaw 추가 (eslint 경고 해결)
  const finalizeCurrentOutline = useCallback(() => {
    const raw = outlineRawPointsRef.current;
    if (!raw || raw.length < 3) {
      alert("윤곽선은 최소 3개 이상 점이 필요합니다.");
      return;
    }
    pushUndoSnapshot();

    const { points: snappedOutline, rawPoints: rawOutline } =
      finalizeOutlineFromRaw(raw, true);
    const id = `outline-${Date.now()}-${Math.random()
      .toString(36)
      .slice(2, 6)}`;

    setElements((prev) => [
      ...prev,
      {
        id,
        type: "건물윤곽",
        points: snappedOutline,
        rawPoints: rawOutline,
        floor: currentFloorIndex,
      },
    ]);

    setOutlinePoints([]);
    setOutlineRawPoints([]);
    setMode(null);
  }, [
    currentFloorIndex,
    finalizeOutlineFromRaw,
    pushUndoSnapshot,
    setElements,
  ]);

  // ====== Hit Test ======
  function pointToSegmentDistance(p, a, b) {
    const vx = b.x - a.x;
    const vy = b.y - a.y;
    const wx = p.x - a.x;
    const wy = p.y - a.y;
    const c1 = vx * wx + vy * wy;
    if (c1 <= 0) return Math.hypot(p.x - a.x, p.y - a.y);
    const c2 = vx * vx + vy * vy;
    if (c2 <= c1) return Math.hypot(p.x - b.x, p.y - b.y);
    const t = c1 / c2;
    const projx = a.x + t * vx;
    const projy = a.y + t * vy;
    return Math.hypot(p.x - projx, p.y - projy);
  }

  const hitTestElement = useCallback(
    (nat) => {
      for (let i = elements.length - 1; i >= 0; i--) {
        const el = elements[i];

        // 다른 층 요소는 무시
        if ((el.floor ?? 0) !== currentFloorIndex) continue;

        if (el.type === "문") {
          const naturalR = 20 / displayedScale;
          const dx = nat.x - el.x;
          const dy = nat.y - el.y;
          if (dx * dx + dy * dy <= naturalR * naturalR)
            return { kind: "door", el };
        } else if (el.type === "비상구") {
          const w = el.width || 30;
          const h = el.height || 30;
          if (
            nat.x >= el.x - w / 2 &&
            nat.x <= el.x + w / 2 &&
            nat.y >= el.y - h / 2 &&
            nat.y <= el.y + h / 2
          )
            return { kind: "exit", el };
        } else if (el.type === "건물윤곽") {
          const pts = el.points || [];
          for (let j = 0; j < pts.length; j++) {
            const a = pts[j];
            const b = pts[(j + 1) % pts.length];
            const dist = pointToSegmentDistance(nat, a, b);
            const threshold = 8 / displayedScale;
            if (dist <= threshold)
              return { kind: "contour-seg", el, segIndex: j, a, b };
          }
        } else {
          if (
            nat.x >= el.x &&
            nat.x <= el.x + el.width &&
            nat.y >= el.y &&
            nat.y <= el.y + el.height
          ) {
            return { kind: "box", el };
          }
        }
      }
      return null;
    },
    [elements, currentFloorIndex, displayedScale]
  );

  // ====== Building bounds (for room clamp / auto adjust) ======
  const getBuildingBounds = useCallback(() => {
    const contours = elements.filter(
      (el) =>
        el.type === "건물윤곽" &&
        (el.floor ?? 0) === currentFloorIndex &&
        el.points?.length
    );

    if (!contours.length) return null;

    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;

    contours.forEach((el) => {
      el.points.forEach((p) => {
        minX = Math.min(minX, p.x);
        minY = Math.min(minY, p.y);
        maxX = Math.max(maxX, p.x);
        maxY = Math.max(maxY, p.y);
      });
    });

    return { minX, minY, maxX, maxY };
  }, [elements, currentFloorIndex]);

  const clampRoomToBuildingBox = useCallback(
    (box) => {
      const bounds = getBuildingBounds();
      if (!bounds) return box;

      let { x, y, width, height } = box;

      if (x < bounds.minX) x = bounds.minX;
      if (y < bounds.minY) y = bounds.minY;
      if (x + width > bounds.maxX) x = bounds.maxX - width;
      if (y + height > bounds.maxY) y = bounds.maxY - height;

      return { ...box, x, y, width, height };
    },
    [getBuildingBounds]
  );

  // 방 자동 보정 (윤곽 바운딩 박스 안으로 + 겹치면 중간선 분할)
  const autoAdjustRooms = useCallback(() => {
    const boundsInfo = getBuildingBounds();
    if (!boundsInfo) {
      alert("건물 윤곽이 있어야 방 보정을 할 수 있습니다.");
      return;
    }

    pushUndoSnapshot();

    setElements((prev) => {
      const rooms = prev
        .filter(
          (el) =>
            (el.floor ?? 0) === currentFloorIndex &&
            (el.type === "방" ||
              el.type === "재난 구역" ||
              el.type === "안전 구역")
        )
        .map((r) => ({ ...r }));

      const doors = prev.filter(
        (el) => (el.floor ?? 0) === currentFloorIndex && el.type === "문"
      );
      const exits = prev.filter(
        (el) => (el.floor ?? 0) === currentFloorIndex && el.type === "비상구"
      );

      const others = prev.filter(
        (el) =>
          (el.floor ?? 0) !== currentFloorIndex ||
          !["방", "재난 구역", "안전 구역", "문", "비상구"].includes(el.type)
      );

      // 1) 일단 바운딩 박스 안으로
      for (let i = 0; i < rooms.length; i++) {
        rooms[i] = clampRoomToBuildingBox(rooms[i]);
      }

      // 2) 겹치면 중간선 분할(간단 정리)
      const n = rooms.length;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const a = rooms[i];
          const b = rooms[j];

          const ax1 = a.x,
            ay1 = a.y,
            ax2 = a.x + a.width,
            ay2 = a.y + a.height;
          const bx1 = b.x,
            by1 = b.y,
            bx2 = b.x + b.width,
            by2 = b.y + b.height;

          const overlapX = Math.min(ax2, bx2) - Math.max(ax1, bx1);
          const overlapY = Math.min(ay2, by2) - Math.max(ay1, by1);
          if (overlapX <= 0 || overlapY <= 0) continue;

          if (overlapX <= overlapY) {
            // 세로로 나누기
            if (ax1 <= bx1) {
              const mid = (ax2 + bx1) / 2;
              a.width = Math.max(1e-3, mid - ax1);
              b.x = mid;
              b.width = Math.max(1e-3, bx2 - mid);
            } else {
              const mid = (bx2 + ax1) / 2;
              b.width = Math.max(1e-3, mid - bx1);
              a.x = mid;
              a.width = Math.max(1e-3, ax2 - mid);
            }
          } else {
            // 가로로 나누기
            if (ay1 <= by1) {
              const mid = (ay2 + by1) / 2;
              a.height = Math.max(1e-3, mid - ay1);
              b.y = mid;
              b.height = Math.max(1e-3, by2 - mid);
            } else {
              const mid = (by2 + ay1) / 2;
              b.height = Math.max(1e-3, mid - by1);
              a.y = mid;
              a.height = Math.max(1e-3, ay2 - mid);
            }
          }
        }
      }

      for (let i = 0; i < rooms.length; i++)
        rooms[i] = clampRoomToBuildingBox(rooms[i]);

      return [...others, ...rooms, ...doors, ...exits];
    });
  }, [
    clampRoomToBuildingBox,
    currentFloorIndex,
    getBuildingBounds,
    pushUndoSnapshot,
    setElements,
  ]);

  // ====== Image upload ======
  const onImageChange = useCallback(
    (e) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const url = URL.createObjectURL(file);
      const floorIdx = currentFloorIndex;

      setFloors((prev) => {
        const next = [...prev];
        const curr = next[floorIdx] || makeEmptyFloor();
        next[floorIdx] = {
          ...curr,
          imageSrc: url,
          uploadedFile: file,
          imgNatural: { w: 0, h: 0 },
          elements: [],
          undoStack: [],
          redoStack: [],
          hasAutoAnalysisResult: false,
          autoElementsCache: [],
        };
        return next;
      });

      // UI reset
      setMode(null);
      setImageLoaded(false);
      setFitScale(1);
      setZoom(1);
      setImgOffset({ x: 0, y: 0 });
      setSelectedId(null);
      setSelectedIds([]);
      setEditingResizeId(null);
      setPreview(null);
      setOutlineRawPoints([]);
      setOutlinePoints([]);
      setContextMenu(null);
      setShowHelp(false);
    },
    [currentFloorIndex, makeEmptyFloor]
  );

  const onImgLoad = useCallback(
    (ev) => {
      const img = ev.target;
      const w = img.naturalWidth || 0;
      const h = img.naturalHeight || 0;

      setFloors((prev) => {
        const next = [...prev];
        const curr = next[currentFloorIndex] || makeEmptyFloor();
        next[currentFloorIndex] = { ...curr, imgNatural: { w, h } };
        return next;
      });

      const fit = w && h ? Math.min(VIEW_W / w, VIEW_H / h, 1) : 1;
      setFitScale(fit);
      setZoom(1);
      setImgOffset({ x: 0, y: 0 });
      setImageLoaded(true);
    },
    [currentFloorIndex, makeEmptyFloor]
  );

  // ====== Floors UI ======
  const goPrevFloor = useCallback(() => {
    setCurrentFloorIndex((p) => Math.max(0, p - 1));
    setMode(null);
    setSelectedId(null);
    setSelectedIds([]);
    setEditingResizeId(null);
    setContextMenu(null);
    setPreview(null);
    setOutlineRawPoints([]);
    setOutlinePoints([]);
    setRotatingDoorId(null);
    setPendingDoorId(null);
  }, []);

  const goNextFloor = useCallback(() => {
    setCurrentFloorIndex((p) => {
      const nextIdx = p + 1;

      setFloorNames((names) => {
        if (nextIdx < names.length) return names;
        return [...names, `${nextIdx + 1}층`];
      });

      setFloors((fs) => {
        if (nextIdx < fs.length) return fs;
        return [...fs, makeEmptyFloor()];
      });

      return nextIdx;
    });

    setMode(null);
    setSelectedId(null);
    setSelectedIds([]);
    setEditingResizeId(null);
    setContextMenu(null);
    setPreview(null);
    setOutlineRawPoints([]);
    setOutlinePoints([]);
    setRotatingDoorId(null);
    setPendingDoorId(null);
  }, [makeEmptyFloor]);

  const renameFloor = useCallback(() => {
    const currName =
      floorNames[currentFloorIndex] || `${currentFloorIndex + 1}층`;
    const name = window.prompt("층 이름을 수정하세요", currName);
    if (!name || !name.trim()) return;

    setFloorNames((prev) => {
      const next = [...prev];
      next[currentFloorIndex] = name.trim();
      return next;
    });
  }, [currentFloorIndex, floorNames]);

  const currentFloorLabel = useMemo(() => {
    return floorNames[currentFloorIndex] || `${currentFloorIndex + 1}층`;
  }, [currentFloorIndex, floorNames]);

  // ====== JSON Save ======
  const handleSaveJSON = useCallback(() => {
    const normalizedElements = (elements || [])
      .filter((el) => (el.floor ?? 0) === currentFloorIndex)
      .map((el) => {
        if (el.type === "건물윤곽") {
          return {
            ...el,
            points: (el.points || []).map((p) => ({
              x: round4(p.x),
              y: round4(p.y),
            })),
            rawPoints: el.rawPoints
              ? el.rawPoints.map((p) => ({ x: round4(p.x), y: round4(p.y) }))
              : undefined,
          };
        }
        if (el.type === "문") {
          return {
            ...el,
            x: round4(el.x),
            y: round4(el.y),
            angle: round4(el.angle || 0),
          };
        }
        return {
          ...el,
          x: round4(el.x),
          y: round4(el.y),
          width: round4(el.width),
          height: round4(el.height),
        };
      });

    const payload = {
      floor: currentFloorLabel,
      image: {
        src: imageSrc,
        natural: { w: round4(imgNatural.w), h: round4(imgNatural.h) },
      },
      elements: normalizedElements,
      uiState: {
        mode,
        showImage,
        zoom: round4(zoom),
      },
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `school-setting-${currentFloorLabel}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [
    currentFloorIndex,
    currentFloorLabel,
    elements,
    imageSrc,
    imgNatural.h,
    imgNatural.w,
    mode,
    showImage,
    zoom,
  ]);

  // ====== Shortcuts (단축키) ======
  const keyGuardRef = useRef({ undo: false, redo: false });

  useEffect(() => {
    const onKeyDown = (e) => {
      // 스페이스 패닝
      if (e.code === "Space") spacePressedRef.current = true;

      // 입력 중이면 단축키 대부분 무시
      const active = document.activeElement;
      const isTyping =
        active &&
        (active.tagName === "INPUT" ||
          active.tagName === "TEXTAREA" ||
          active.tagName === "SELECT" ||
          active.isContentEditable);

      // 방향키 미세이동(선택 모드, 입력중X)
      if (
        !e.ctrlKey &&
        ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)
      ) {
        if (isTyping) return;
        const ids = selectedIdsRef.current;
        if (!ids.length) return;

        e.preventDefault();
        if (!e.repeat) pushUndoSnapshot();

        const step = 1;
        let dx = 0,
          dy = 0;
        if (e.key === "ArrowUp") dy = -step;
        if (e.key === "ArrowDown") dy = step;
        if (e.key === "ArrowLeft") dx = -step;
        if (e.key === "ArrowRight") dx = step;

        setElements((prev) =>
          prev.map((p) => {
            if ((p.floor ?? 0) !== currentFloorIndex) return p;
            if (!ids.includes(p.id)) return p;

            const w = p.width || 0;
            const h = p.height || 0;

            let nx = p.x + dx;
            let ny = p.y + dy;

            // 문/비상구는 중심점 기반이라 간단 clamp
            if (p.type === "문" || p.type === "비상구") {
              nx = clamp(nx, 0, imgNatural.w);
              ny = clamp(ny, 0, imgNatural.h);
              return { ...p, x: nx, y: ny };
            }

            nx = clamp(nx, 0, imgNatural.w - w);
            ny = clamp(ny, 0, imgNatural.h - h);
            return { ...p, x: nx, y: ny };
          })
        );
        return;
      }

      // Delete/Backspace 삭제
      if (!e.ctrlKey && (e.key === "Backspace" || e.key === "Delete")) {
        if (isTyping) return;
        const ids = selectedIdsRef.current;
        if (!ids.length) return;

        e.preventDefault();
        pushUndoSnapshot();
        setElements((prev) => prev.filter((p) => !ids.includes(p.id)));
        setSelectedId(null);
        setSelectedIds([]);
        setEditingResizeId(null);
        return;
      }

      // Ctrl 조합
      if (!e.ctrlKey) return;
      if (e.repeat) return;

      // Ctrl+Z (윤곽 그리기 중이면 점 하나 되돌리기)
      if (e.key === "z" || e.key === "Z") {
        const isOutlineMode = modeRef.current === "건물윤곽";
        if (isOutlineMode && outlineRawPointsRef.current.length > 0) {
          e.preventDefault();
          setOutlineRawPoints((prev) => prev.slice(0, -1));
          setOutlinePoints((prev) => prev.slice(0, -1));
          return;
        }

        if (keyGuardRef.current.undo) return;
        keyGuardRef.current.undo = true;
        e.preventDefault();
        performUndoOnce();
        setSelectedId(null);
        setSelectedIds([]);
        setEditingResizeId(null);
        return;
      }

      // Ctrl+Y
      if (e.key === "y" || e.key === "Y") {
        if (keyGuardRef.current.redo) return;
        keyGuardRef.current.redo = true;
        e.preventDefault();
        performRedoOnce();
        setSelectedId(null);
        setSelectedIds([]);
        setEditingResizeId(null);
        return;
      }

      const idsNow = selectedIdsRef.current;
      const selectedElements = elements.filter((p) => idsNow.includes(p.id));

      // Ctrl+X
      if (e.key === "x" || e.key === "X") {
        if (!selectedElements.length) return;
        e.preventDefault();
        pushUndoSnapshot();
        setElements((prev) => prev.filter((p) => !idsNow.includes(p.id)));
        setClipboard(JSON.parse(JSON.stringify(selectedElements)));
        setSelectedId(null);
        setSelectedIds([]);
        setEditingResizeId(null);
        return;
      }

      // Ctrl+C
      if (e.key === "c" || e.key === "C") {
        if (!selectedElements.length) return;
        e.preventDefault();
        setClipboard(JSON.parse(JSON.stringify(selectedElements)));
        return;
      }

      // Ctrl+V
      if (e.key === "v" || e.key === "V") {
        const data = clipboardRef.current;
        if (!data || !Array.isArray(data) || !data.length) return;

        e.preventDefault();
        pushUndoSnapshot();

        const now = Date.now();
        const copies = data.map((el, idx) => ({
          ...el,
          id: `${now}-${idx}-${Math.random().toString(36).slice(2, 6)}`,
          floor: currentFloorIndex,
          x: typeof el.x === "number" ? el.x + 5 : el.x,
          y: typeof el.y === "number" ? el.y + 5 : el.y,
        }));

        setElements((prev) => [...prev, ...copies]);

        const newIds = copies.map((c) => c.id);
        setSelectedIds(newIds);
        setSelectedId(newIds[newIds.length - 1]);
        return;
      }

      // Ctrl+0 : 줌 리셋
      if (e.key === "0") {
        e.preventDefault();
        if (!imageSrc || !imageLoaded) return;
        setZoom(1);
        setImgOffset({ x: 0, y: 0 });
        return;
      }

      // Ctrl+'+' / Ctrl+'-'
      if (e.key === "=" || e.key === "+") {
        e.preventDefault();
        zoomIn();
        return;
      }
      if (e.key === "-" || e.key === "_") {
        e.preventDefault();
        zoomOut();
        return;
      }
    };

    const onKeyUp = (e) => {
      if (e.code === "Space") spacePressedRef.current = false;
      if (e.key === "z" || e.key === "Z") keyGuardRef.current.undo = false;
      if (e.key === "y" || e.key === "Y") keyGuardRef.current.redo = false;
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [
    currentFloorIndex,
    elements,
    imgNatural.h,
    imgNatural.w,
    imageLoaded,
    imageSrc,
    performRedoOnce,
    performUndoOnce,
    pushUndoSnapshot,
    setElements,
    zoomIn,
    zoomOut,
    clamp,
  ]);

  // ESC
  useEffect(() => {
    const fn = (ev) => {
      if (ev.key !== "Escape") return;

      // 윤곽 모드에서 점이 적으면 초기화
      if (
        modeRef.current === "건물윤곽" &&
        outlineRawPointsRef.current.length < 3
      ) {
        setOutlinePoints([]);
        setOutlineRawPoints([]);
      }

      setContextMenu(null);
      setRotatingDoorId(null);
      setResizing(null);
      setMode(null);
      setIsPanning(false);
      setIsBoxSelecting(false);
      setSelectionRect(null);
      setEditingResizeId(null);
      setPendingDoorId(null);
      setPreview(null);
      setIsDrawing(false);
      setDrawStart(null);
    };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, []);

  // ====== Mouse interactions ======

  // 그룹 드래그(요소 이동)
  const [dragging, setDragging] = useState(null);
  // dragging = { type: "elements", startNat, ids, origPositions }

  const startElementsDrag = useCallback(
    (ids, startNat) => {
      if (!ids?.length) return;
      const origPositions = {};
      elements.forEach((p) => {
        if (ids.includes(p.id)) origPositions[p.id] = { x: p.x, y: p.y };
      });
      setDragging({ type: "elements", startNat, ids: [...ids], origPositions });
    },
    [elements]
  );

  // 마우스 Down
  const onViewportMouseDown = useCallback(
    (e) => {
      if (e.button === 2) return; // 우클릭은 contextMenu에서 처리
      if (!viewportRef.current) return;
      if (!imageSrc || !imageLoaded) return;

      const rect = viewportRef.current.getBoundingClientRect();
      if (
        e.clientX < rect.left ||
        e.clientX > rect.right ||
        e.clientY < rect.top ||
        e.clientY > rect.bottom
      )
        return;

      const nat = clientToNatural(e.clientX, e.clientY);
      const inside = isInsideImageNatural(nat);
      const clicked = inside ? hitTestElement(nat) : null;

      // 문 회전 확정
      if (rotatingDoorId && mode === "문") {
        setRotatingDoorId(null);
        return;
      }
      if (resizing) return;
      if (rotatingDoorId) return;

      // ===== 선택 모드 =====
      if (mode === null) {
        // Space+드래그 = 패닝
        if (
          !clicked &&
          inside &&
          spacePressedRef.current &&
          (displayedW > VIEW_W || displayedH > VIEW_H)
        ) {
          setIsPanning(true);
          panStartRef.current = { x: e.clientX, y: e.clientY };
          panOriginRef.current = { ...imgOffsetRef.current };
          setContextMenu(null);
          return;
        }

        // 요소 클릭 선택
        if (clicked) {
          const elId = clicked.el.id;
          const current = selectedIdsRef.current;
          let newSelectedIds;

          if (e.ctrlKey) {
            newSelectedIds = current.includes(elId)
              ? current.filter((id) => id !== elId)
              : [...current, elId];
          } else {
            newSelectedIds = [elId];
          }

          setSelectedIds(newSelectedIds);
          setSelectedId(
            newSelectedIds.length
              ? newSelectedIds[newSelectedIds.length - 1]
              : null
          );
          setContextMenu(null);
          setEditingResizeId(null);

          // 바로 드래그 시작
          startElementsDrag(newSelectedIds, nat);
          return;
        }

        // 빈 공간 = 박스 선택 시작
        setSelectedId(null);
        setSelectedIds([]);
        setDragging(null);
        setEditingResizeId(null);

        if (inside) {
          setIsBoxSelecting(true);
          selectionStartRef.current = { x: e.clientX, y: e.clientY };
          setSelectionRect({
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
            w: 0,
            h: 0,
          });
        }
        setContextMenu(null);
        return;
      }

      // ===== 문 모드 =====
      if (mode === "문" && inside) {
        pushUndoSnapshot();
        const id = `door-${Date.now()}-${Math.random()
          .toString(36)
          .slice(2, 6)}`;
        const newDoor = {
          id,
          type: "문",
          x: nat.x,
          y: nat.y,
          angle: 0,
          floor: currentFloorIndex,
        };
        setElements((prev) => [...prev, newDoor]);
        setSelectedId(id);
        setSelectedIds([id]);
        setPendingDoorId(id);
        return;
      }

      // ===== 비상구 모드 =====
      if (mode === "비상구" && inside) {
        pushUndoSnapshot();
        const id = `exit-${Date.now()}-${Math.random()
          .toString(36)
          .slice(2, 6)}`;
        const size = 40;
        const newExit = {
          id,
          type: "비상구",
          x: nat.x,
          y: nat.y,
          width: size,
          height: size,
          floor: currentFloorIndex,
        };
        setElements((prev) => [...prev, newExit]);
        setSelectedId(id);
        setSelectedIds([id]);
        return;
      }

      // ===== 건물 윤곽 모드 =====
      if (mode === "건물윤곽" && inside) {
        const naturalSnapClose = 8 / displayedScale;

        if (outlineRawPoints.length === 0) {
          setOutlineRawPoints([nat]);
          setOutlinePoints([nat]);
        } else {
          const first = outlineRawPoints[0];
          const dxClose = nat.x - first.x;
          const dyClose = nat.y - first.y;
          if (
            Math.hypot(dxClose, dyClose) <= naturalSnapClose &&
            outlineRawPoints.length >= 3
          ) {
            finalizeCurrentOutline();
          } else {
            setOutlineRawPoints((prev) => [...prev, nat]);
            setOutlinePoints((prev) => [...prev, nat]);
          }
        }
        return;
      }

      // ===== 박스 모드 =====
      if (["제한 구역", "방", "재난 구역", "안전 구역"].includes(mode)) {
        if (mode !== "제한 구역" && !inside) return;

        const natClamped = {
          x: clamp(nat.x, 0, imgNatural.w),
          y: clamp(nat.y, 0, imgNatural.h),
        };

        setIsDrawing(true);
        setDrawStart(mode === "제한 구역" ? natClamped : nat);
        setPreview(null);
        return;
      }
    },
    [
      clamp,
      clientToNatural,
      displayedH,
      displayedScale,
      displayedW,
      finalizeCurrentOutline,
      hitTestElement,
      imageLoaded,
      imageSrc,
      imgNatural.h,
      imgNatural.w,
      isInsideImageNatural,
      mode,
      outlineRawPoints,
      pushUndoSnapshot,
      resizing,
      rotatingDoorId,
      currentFloorIndex,
      setElements,
      startElementsDrag,
    ]
  );

  // 마우스 Move
  const onViewportMouseMove = useCallback(
    (e) => {
      if (!imageSrc || !imageLoaded) return;
      if (!viewportRef.current) return;

      const rect = viewportRef.current.getBoundingClientRect();
      const nat = clientToNatural(e.clientX, e.clientY);
      const inside = isInsideImageNatural(nat);

      // hover
      const hitForHover = inside ? hitTestElement(nat) : null;
      if (hitForHover) {
        if (hitForHover.kind === "contour-seg") {
          setHover({
            kind: "contour-seg",
            id: hitForHover.el.id,
            segIndex: hitForHover.segIndex,
            el: hitForHover.el,
          });
        } else if (hitForHover.kind === "box")
          setHover({ kind: "box", id: hitForHover.el.id });
        else if (hitForHover.kind === "door")
          setHover({ kind: "door", id: hitForHover.el.id });
        else if (hitForHover.kind === "exit")
          setHover({ kind: "exit", id: hitForHover.el.id });
      } else {
        setHover(null);
      }

      // 패닝
      if (isPanning && panStartRef.current && panOriginRef.current) {
        const dx = e.clientX - panStartRef.current.x;
        const dy = e.clientY - panStartRef.current.y;

        const newScale = fitScale * zoom;
        const desiredX = panOriginRef.current.x + dx;
        const desiredY = panOriginRef.current.y + dy;

        const clampedOffset = clampOffsetForScale(desiredX, desiredY, newScale);
        setImgOffset(clampedOffset);
        return;
      }

      // 박스 선택 드래그
      if (isBoxSelecting && selectionStartRef.current) {
        const sx = selectionStartRef.current.x - rect.left;
        const sy = selectionStartRef.current.y - rect.top;
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;

        const x = Math.min(sx, cx);
        const y = Math.min(sy, cy);
        const w = Math.abs(cx - sx);
        const h = Math.abs(cy - sy);
        setSelectionRect({ x, y, w, h });
        return;
      }

      // 리사이즈
      if (resizing) {
        const { id, handle, startNat, orig } = resizing;
        const dx = nat.x - startNat.x;
        const dy = nat.y - startNat.y;

        let nx = orig.x;
        let ny = orig.y;
        let nw = orig.width;
        let nh = orig.height;

        if (handle.includes("n")) {
          ny = orig.y + dy;
          nh = orig.height - dy;
        }
        if (handle.includes("s")) nh = orig.height + dy;
        if (handle.includes("w")) {
          nx = orig.x + dx;
          nw = orig.width - dx;
        }
        if (handle.includes("e")) nw = orig.width + dx;

        const minSize = 5 / displayedScale;
        if (nw < minSize) nw = minSize;
        if (nh < minSize) nh = minSize;

        nx = clamp(nx, 0, imgNatural.w - nw);
        ny = clamp(ny, 0, imgNatural.h - nh);

        setElements((prev) =>
          prev.map((p) =>
            p.id === id ? { ...p, x: nx, y: ny, width: nw, height: nh } : p
          )
        );
        return;
      }

      // 문 회전(45도 스냅)
      if (rotatingDoorId) {
        setElements((prev) =>
          prev.map((p) => {
            if (p.id !== rotatingDoorId) return p;
            const dx = nat.x - p.x;
            const dy = nat.y - p.y;
            let angle = (Math.atan2(dy, dx) * 180) / Math.PI;
            const snapped = Math.round(angle / 45) * 45;
            return { ...p, angle: snapped };
          })
        );
        return;
      }

      // 요소 그룹 드래그
      if (dragging && dragging.type === "elements" && mode === null && inside) {
        const dx = nat.x - dragging.startNat.x;
        const dy = nat.y - dragging.startNat.y;

        setElements((prev) =>
          prev.map((p) => {
            if ((p.floor ?? 0) !== currentFloorIndex) return p;
            if (!dragging.ids.includes(p.id)) return p;

            const orig = dragging.origPositions[p.id];
            if (!orig) return p;

            if (p.type === "문" || p.type === "비상구") {
              const nx = clamp(orig.x + dx, 0, imgNatural.w);
              const ny = clamp(orig.y + dy, 0, imgNatural.h);
              return { ...p, x: nx, y: ny };
            }

            const w = p.width || 0;
            const h = p.height || 0;
            const nx = clamp(orig.x + dx, 0, imgNatural.w - w);
            const ny = clamp(orig.y + dy, 0, imgNatural.h - h);
            return { ...p, x: nx, y: ny };
          })
        );
        return;
      }

      // 박스 미리보기
      if (isDrawing && drawStart) {
        if (mode !== "제한 구역" && !inside) return;

        const natNow =
          mode === "제한 구역"
            ? {
                x: clamp(nat.x, 0, imgNatural.w),
                y: clamp(nat.y, 0, imgNatural.h),
              }
            : nat;

        let rectNat = {
          x: Math.max(0, Math.min(drawStart.x, natNow.x)),
          y: Math.max(0, Math.min(drawStart.y, natNow.y)),
          width: Math.min(imgNatural.w, Math.abs(natNow.x - drawStart.x)),
          height: Math.min(imgNatural.h, Math.abs(natNow.y - drawStart.y)),
        };

        if (mode === "방") rectNat = clampRoomToBuildingBox(rectNat);
        setPreview(rectNat);
        return;
      }
    },
    [
      clamp,
      clampOffsetForScale,
      clampRoomToBuildingBox,
      clientToNatural,
      displayedScale,
      dragging,
      drawStart,
      fitScale,
      hitTestElement,
      imageLoaded,
      imageSrc,
      imgNatural.h,
      imgNatural.w,
      isBoxSelecting,
      isDrawing,
      isInsideImageNatural,
      isPanning,
      mode,
      resizing,
      rotatingDoorId,
      zoom,
      currentFloorIndex,
      setElements,
    ]
  );

  // 마우스 Up
  const onViewportMouseUp = useCallback(() => {
    // 문: 찍고 나서 각도 조정 시작
    if (pendingDoorId) {
      setRotatingDoorId(pendingDoorId);
      setPendingDoorId(null);
    }

    if (isPanning) setIsPanning(false);

    // 박스 선택 종료
    if (isBoxSelecting && selectionRect && viewportRef.current) {
      const vpRect = viewportRef.current.getBoundingClientRect();
      const toNatural = (sx, sy) =>
        clientToNatural(vpRect.left + sx, vpRect.top + sy);

      const p1 = toNatural(selectionRect.x, selectionRect.y);
      const p2 = toNatural(
        selectionRect.x + selectionRect.w,
        selectionRect.y + selectionRect.h
      );

      const minX = Math.min(p1.x, p2.x);
      const maxX = Math.max(p1.x, p2.x);
      const minY = Math.min(p1.y, p2.y);
      const maxY = Math.max(p1.y, p2.y);

      const ids = [];
      elements.forEach((el) => {
        if ((el.floor ?? 0) !== currentFloorIndex) return;

        if (el.type === "문") {
          if (el.x >= minX && el.x <= maxX && el.y >= minY && el.y <= maxY)
            ids.push(el.id);
        } else if (el.type === "비상구") {
          const w = el.width || 30;
          const h = el.height || 30;
          const cx = el.x,
            cy = el.y;
          if (
            cx + w / 2 >= minX &&
            cx - w / 2 <= maxX &&
            cy + h / 2 >= minY &&
            cy - h / 2 <= maxY
          )
            ids.push(el.id);
        } else if (el.type !== "건물윤곽") {
          const ex1 = el.x,
            ex2 = el.x + el.width;
          const ey1 = el.y,
            ey2 = el.y + el.height;
          if (ex2 >= minX && ex1 <= maxX && ey2 >= minY && ey1 <= maxY)
            ids.push(el.id);
        }
      });

      setSelectedIds(ids);
      setSelectedId(ids.length ? ids[ids.length - 1] : null);
      setIsBoxSelecting(false);
      setSelectionRect(null);
    }

    // 리사이즈 종료
    if (resizing) setResizing(null);

    // 박스 생성 완료
    if (isDrawing && preview) {
      if (mode === "방") {
        const name = window.prompt("이 방의 이름을 입력하세요:", "");
        if (name === null) {
          setIsDrawing(false);
          setDrawStart(null);
          setPreview(null);
          setDragging(null);
          return;
        }

        const id = `room-${Date.now()}-${Math.random()
          .toString(36)
          .slice(2, 6)}`;
        pushUndoSnapshot();
        const rectNat = clampRoomToBuildingBox(preview);

        const newBox = {
          id,
          type: "방",
          ...rectNat,
          name,
          floor: currentFloorIndex,
        };
        setElements((prev) => [...prev, newBox]);
        setSelectedId(id);
        setSelectedIds([id]);
      } else {
        const id = `box-${Date.now()}-${Math.random()
          .toString(36)
          .slice(2, 6)}`;
        pushUndoSnapshot();
        const newBox = { id, type: mode, ...preview, floor: currentFloorIndex };
        setElements((prev) => [...prev, newBox]);
        setSelectedId(id);
        setSelectedIds([id]);
      }
    }

    setIsDrawing(false);
    setDrawStart(null);
    setPreview(null);
    setDragging(null);
  }, [
    clampRoomToBuildingBox,
    clientToNatural,
    currentFloorIndex,
    elements,
    isBoxSelecting,
    isDrawing,
    isPanning,
    mode,
    pendingDoorId,
    preview,
    pushUndoSnapshot,
    resizing,
    selectionRect,
    setElements,
  ]);

  // 컨텍스트 메뉴(우클릭)
  const onViewportContextMenu = useCallback(
    (e) => {
      e.preventDefault();
      if (!imageSrc || !imageLoaded) return;

      const nat = clientToNatural(e.clientX, e.clientY);
      const hit = isInsideImageNatural(nat) ? hitTestElement(nat) : null;

      // 빈 공간
      if (!hit) {
        const opts = [
          {
            label: "방 자동 보정(윤곽 바운딩 박스/겹침 정리)",
            action: () => {
              autoAdjustRooms();
              setContextMenu(null);
            },
          },
          {
            label: "Undo (되돌리기)",
            action: () => {
              performUndoOnce();
              setContextMenu(null);
              setSelectedId(null);
              setSelectedIds([]);
              setEditingResizeId(null);
            },
          },
          {
            label: "Redo (다시하기)",
            action: () => {
              performRedoOnce();
              setContextMenu(null);
              setSelectedId(null);
              setSelectedIds([]);
              setEditingResizeId(null);
            },
          },
        ];

        const flipY = e.clientY > window.innerHeight - 220;
        setContextMenu({ x: e.clientX, y: e.clientY, flipY, options: opts });
        return;
      }

      // 요소 우클릭: 선택
      setSelectedId(hit.el.id);
      setSelectedIds([hit.el.id]);

      const opts = [];

      // 건물윤곽은 삭제만(간단)
      if (hit.el.type !== "건물윤곽") {
        opts.push({
          label: "삭제",
          action: () => {
            pushUndoSnapshot();
            setElements((prev) => prev.filter((p) => p.id !== hit.el.id));
            setContextMenu(null);
            setSelectedId(null);
            setSelectedIds([]);
            setEditingResizeId(null);
          },
        });
      } else {
        opts.push({
          label: "윤곽 전체 삭제",
          action: () => {
            pushUndoSnapshot();
            setElements((prev) => prev.filter((p) => p.id !== hit.el.id));
            setContextMenu(null);
            setSelectedId(null);
            setSelectedIds([]);
          },
        });
      }

      // 제한구역: 리사이즈 모드
      if (hit.el.type === "제한 구역") {
        opts.push({
          label: "형태 수정(리사이즈)",
          action: () => {
            setMode(null);
            setEditingResizeId(hit.el.id);
            setContextMenu(null);
          },
        });
      }

      // 방/재난/안전: 타입 변경 + 이름 수정 + 리사이즈
      if (["방", "재난 구역", "안전 구역"].includes(hit.el.type)) {
        opts.push(
          {
            label: "방(파란색)으로 변경",
            action: () => {
              pushUndoSnapshot();
              setElements((prev) =>
                prev.map((p) => (p.id === hit.el.id ? { ...p, type: "방" } : p))
              );
              setContextMenu(null);
            },
          },
          {
            label: "재난 구역(빨간색)으로 변경",
            action: () => {
              pushUndoSnapshot();
              setElements((prev) =>
                prev.map((p) =>
                  p.id === hit.el.id ? { ...p, type: "재난 구역" } : p
                )
              );
              setContextMenu(null);
            },
          },
          {
            label: "안전 구역(초록색)으로 변경",
            action: () => {
              pushUndoSnapshot();
              setElements((prev) =>
                prev.map((p) =>
                  p.id === hit.el.id ? { ...p, type: "안전 구역" } : p
                )
              );
              setContextMenu(null);
            },
          },
          {
            label: "방 이름 수정",
            action: () => {
              const curr = hit.el.name || "";
              const name = window.prompt("새 방 이름을 입력하세요", curr);
              if (name !== null) {
                pushUndoSnapshot();
                setElements((prev) =>
                  prev.map((p) => (p.id === hit.el.id ? { ...p, name } : p))
                );
              }
              setContextMenu(null);
            },
          },
          {
            label: "형태 수정(리사이즈)",
            action: () => {
              setMode(null);
              setEditingResizeId(hit.el.id);
              setContextMenu(null);
            },
          }
        );
      }

      // 문: 180도 + 각도 드래그
      if (hit.el.type === "문") {
        opts.push(
          {
            label: "문 반전(180° 회전)",
            action: () => {
              pushUndoSnapshot();
              setElements((prev) =>
                prev.map((p) => {
                  if (p.id !== hit.el.id) return p;
                  return { ...p, angle: ((p.angle || 0) + 180) % 360 };
                })
              );
              setContextMenu(null);
            },
          },
          {
            label: "각도 수정(드래그, 45° 스냅)",
            action: () => {
              setRotatingDoorId(hit.el.id);
              setContextMenu(null);
            },
          }
        );
      }

      const flipY = e.clientY > window.innerHeight - 220;
      setContextMenu({ x: e.clientX, y: e.clientY, flipY, options: opts });
    },
    [
      autoAdjustRooms,
      clientToNatural,
      imageLoaded,
      imageSrc,
      isInsideImageNatural,
      hitTestElement,
      performRedoOnce,
      performUndoOnce,
      pushUndoSnapshot,
      setElements,
    ]
  );

  // 메뉴 밖 클릭 시 닫기 + 화면 밖 클릭 시 선택 해제
  useEffect(() => {
    function onDocMouseDown(e) {
      if (contextMenu && menuRef.current && !menuRef.current.contains(e.target))
        setContextMenu(null);

      if (
        viewportRef.current &&
        !viewportRef.current.contains(e.target) &&
        (!menuRef.current || !menuRef.current.contains(e.target))
      ) {
        setSelectedId(null);
        setSelectedIds([]);
        setEditingResizeId(null);
      }
    }
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [contextMenu]);

  // ====== Resize handles ======
  const handleSize = 8;
  const handleCursorMap = {
    nw: "nwse-resize",
    se: "nwse-resize",
    ne: "nesw-resize",
    sw: "nesw-resize",
    n: "ns-resize",
    s: "ns-resize",
    e: "ew-resize",
    w: "ew-resize",
  };

  function renderResizeHandles(el) {
    if (!el) return null;
    if (!["방", "제한 구역", "재난 구역", "안전 구역"].includes(el.type))
      return null;
    if (mode !== null) return null;
    if (editingResizeId !== el.id) return null;

    const dispW = Math.max(1, natToDispW(el.width));
    const dispH = Math.max(1, natToDispH(el.height));

    const handles = {
      nw: { x: 0, y: 0 },
      n: { x: dispW / 2, y: 0 },
      ne: { x: dispW, y: 0 },
      e: { x: dispW, y: dispH / 2 },
      se: { x: dispW, y: dispH },
      s: { x: dispW / 2, y: dispH },
      sw: { x: 0, y: dispH },
      w: { x: 0, y: dispH / 2 },
    };

    return Object.entries(handles).map(([name, pt]) => (
      <div
        key={name}
        onMouseDown={(ev) => {
          ev.stopPropagation();
          ev.preventDefault();
          pushUndoSnapshot();

          const startNat = clientToNatural(ev.clientX, ev.clientY);
          setResizing({
            id: el.id,
            handle: name,
            startNat,
            orig: { x: el.x, y: el.y, width: el.width, height: el.height },
          });

          setContextMenu(null);
          setDragging(null);
        }}
        style={{
          position: "absolute",
          left: pt.x - handleSize / 2,
          top: pt.y - handleSize / 2,
          width: handleSize,
          height: handleSize,
          background: "#fff",
          border: "1px solid #444",
          borderRadius: 2,
          zIndex: 200,
          cursor: handleCursorMap[name] || "move",
          pointerEvents: "auto",
          boxSizing: "border-box",
        }}
      />
    ));
  }

  // ====== UI classes ======
  const btnSub =
    "px-4 py-2 bg-[#66BB6A] text-white rounded-lg shadow hover:bg-[#2E7D32] disabled:opacity-60";
  const btnGray =
    "px-4 py-2 bg-gray-700 text-white rounded-lg shadow hover:bg-gray-800 disabled:opacity-60";
  const chipActive = "bg-[#E8F5E9] border-[#2E7D32] text-[#2E7D32]";
  const chipIdle = "bg-white border-gray-200 text-gray-700 hover:bg-gray-50";

  // ====== Render ======
  return (
    <div className="bg-[#F9FBE7] min-h-screen overflow-x-hidden">
      <Navbar />

      <div className="p-8">
        <h2 className="text-3xl font-bold text-[#2E7D32]">학교 환경 설정</h2>

        {/* 상단 툴바 */}
        <div className="p-4 bg-white rounded shadow space-y-4">
          <div className="flex flex-wrap gap-3 items-center justify-between">
            <div className="flex flex-wrap gap-3 items-center">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={onImageChange}
                style={{ display: "none" }}
              />

              <button
                onClick={() => fileInputRef.current?.click()}
                className={btnGray}
              >
                이미지 업로드
              </button>

              <button
                onClick={() => setShowImage((p) => !p)}
                className={btnGray}
                disabled={!imageSrc}
              >
                {showImage ? "이미지 숨기기" : "이미지 보이기"}
              </button>

              <button onClick={() => setShowHelp((p) => !p)} className={btnSub}>
                {showHelp ? "사용 방법 닫기" : "사용 방법"}
              </button>

              <button
                onClick={autoAdjustRooms}
                className={btnSub}
                disabled={!imageSrc}
              >
                방 자동 보정
              </button>
            </div>

            {/* 층 이동/이름 */}
            <div className="flex items-center gap-2">
              <button
                className="px-3 py-2 border rounded-lg text-sm bg-white hover:bg-gray-50 disabled:opacity-60"
                onClick={goPrevFloor}
                disabled={currentFloorIndex === 0}
              >
                ◀ 이전 층
              </button>

              <div className="px-4 py-2 bg-[#E8F5E9] rounded-lg border border-[#2E7D32] text-sm font-medium flex items-center gap-2">
                <span className="text-[#2E7D32]">{currentFloorLabel}</span>
                <button
                  className="text-xs px-2 py-1 border rounded bg-white"
                  onClick={renameFloor}
                >
                  ✏
                </button>
              </div>

              <button
                className="px-3 py-2 border rounded-lg text-sm bg-white hover:bg-gray-50"
                onClick={goNextFloor}
              >
                다음 층 ▶
              </button>
            </div>
          </div>

          {/* 모드 버튼 */}
          <div className="flex flex-wrap gap-2 items-center">
            <button
              onClick={handleAutoAnalyze}
              className={btnSub}
              disabled={!imageSrc || autoAnalyzing}
            >
              {autoAnalyzing ? "자동 분석 중..." : "구조도 자동 분석"}
            </button>
            <div className="text-sm font-semibold text-gray-700 mr-2">
              도구:
            </div>

            {modeButtons.map((m) => (
              <button
                key={m}
                onClick={() => {
                  setMode((prev) => {
                    if (prev === m) {
                      if (m === "건물윤곽") {
                        setOutlinePoints([]);
                        setOutlineRawPoints([]);
                      }
                      return null;
                    }
                    return m;
                  });
                  setContextMenu(null);
                  setEditingResizeId(null);
                }}
                className={`px-3 py-2 rounded-full border text-sm ${
                  mode === m ? chipActive : chipIdle
                }`}
                disabled={!imageSrc}
              >
                {m}
              </button>
            ))}

            <div className="ml-auto flex items-center gap-2 text-sm">
              <button
                onClick={zoomOut}
                className="px-3 py-2 border rounded bg-white"
                disabled={!imageSrc}
              >
                -
              </button>
              <div className="text-gray-700 w-20 text-center">
                {zoom.toFixed(2)}×
              </div>
              <button
                onClick={zoomIn}
                className="px-3 py-2 border rounded bg-white"
                disabled={!imageSrc}
              >
                +
              </button>
              <button
                onClick={() => {
                  setZoom(1);
                  setImgOffset({ x: 0, y: 0 });
                }}
                className="px-3 py-2 border rounded bg-white"
                disabled={!imageSrc}
              >
                리셋
              </button>
              <button
                onClick={handleSaveJSON}
                disabled={!imageSrc}
                className="
        px-4 py-2
        rounded
        bg-[#2E7D32]
        text-white
        font-bold
        shadow
        hover:bg-[#256428]
        disabled:opacity-60
      "
              >
                저장
              </button>
            </div>
          </div>

          {/* 도움말 */}
          {showHelp && (
            <div className="p-4 rounded border bg-[#FAFAFA] text-sm text-gray-800 space-y-3">
              <div className="font-bold text-[#2E7D32]">
                사용 방법(핵심 조작)
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="p-3 bg-white border rounded">
                  <div className="font-semibold mb-2">마우스 조작</div>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>
                      <b>선택 모드</b>(도구 선택 안함): 요소 클릭=선택,{" "}
                      <b>Ctrl+클릭</b>=다중 선택
                    </li>
                    <li>빈 공간 드래그=박스 선택(여러 요소 선택)</li>
                    <li>
                      <b>Space 누른 채 드래그</b>=패닝(이미지 이동)
                    </li>
                    <li>
                      우클릭=컨텍스트 메뉴(삭제/타입변경/리사이즈/문 회전 등)
                    </li>
                    <li>
                      문은 <b>문 모드</b>에서 클릭으로 생성 후, 마우스로 각도
                      조절(45° 스냅)
                    </li>
                    <li>
                      윤곽은 <b>건물윤곽 모드</b>에서 점을 찍고, 시작점 근처
                      클릭하면 닫힘
                    </li>
                  </ul>
                </div>

                <div className="p-3 bg-white border rounded">
                  <div className="font-semibold mb-2">단축키</div>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>
                      <b>Ctrl+Z</b>: Undo (윤곽 그리기 중이면 점 1개 취소)
                    </li>
                    <li>
                      <b>Ctrl+Y</b>: Redo
                    </li>
                    <li>
                      <b>Delete / Backspace</b>: 선택 요소 삭제
                    </li>
                    <li>
                      <b>Ctrl+C / Ctrl+X / Ctrl+V</b>: 복사 / 잘라내기 /
                      붙여넣기
                    </li>
                    <li>
                      <b>방향키</b>: 선택 요소 1px 이동
                    </li>
                    <li>
                      <b>Ctrl + 마우스 휠</b>: 줌 인/아웃(커서 기준)
                    </li>
                    <li>
                      <b>일반 휠</b>: 세로 스크롤(이미지가 세로로 넘칠 때)
                    </li>
                    <li>
                      <b>Ctrl+0</b>: 줌/오프셋 리셋
                    </li>
                    <li>
                      <b>ESC</b>: 현재 모드/드래그/리사이즈/회전/프리뷰 취소
                    </li>
                  </ul>
                </div>
              </div>

              <div className="text-xs text-gray-600">
                팁) 리사이즈는 요소를 우클릭 → “형태 수정(리사이즈)”를 누르면
                핸들이 나타납니다.
              </div>
            </div>
          )}
        </div>

        {/* ✅ 편집 영역: 뷰포트(왼쪽) + 상태 패널(오른쪽) */}
        <div className="flex gap-4 items-start mt-4">
          {/* 뷰포트 */}
          <div
            ref={viewportRef}
            className="relative bg-white shadow border"
            style={{
              width: VIEW_W,
              height: VIEW_H,
              userSelect: "none",
              overflow: "hidden",
              position: "relative",
            }}
            onMouseDown={onViewportMouseDown}
            onMouseMove={onViewportMouseMove}
            onMouseUp={onViewportMouseUp}
            onContextMenu={onViewportContextMenu}
            onWheel={onViewportWheel}
          >
            {/* 이미지 */}
            {imageSrc && showImage ? (
              <img
                src={imageSrc}
                alt="floorplan"
                onLoad={onImgLoad}
                draggable={false}
                style={{
                  position: "absolute",
                  left: imgLeft,
                  top: imgTop,
                  width: displayedW,
                  height: displayedH,
                  pointerEvents: "none",
                }}
              />
            ) : !imageSrc ? (
              <div
                style={{
                  width: "100%",
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#777",
                }}
              >
                이미지를 업로드하세요 (이 박스 안에서만 편집)
              </div>
            ) : (
              <div
                style={{
                  color: "#777",
                  position: "absolute",
                  left: 12,
                  top: 12,
                }}
              >
                이미지 숨김 상태
              </div>
            )}

            {/* 건물 윤곽 SVG */}
            <svg
              style={{
                position: "absolute",
                left: imgLeft,
                top: imgTop,
                width: displayedW,
                height: displayedH,
                pointerEvents: "none",
                zIndex: 45,
              }}
            >
              {/* 채움 */}
              {elements
                .filter(
                  (el) =>
                    el.type === "건물윤곽" &&
                    (el.floor ?? 0) === currentFloorIndex
                )
                .map((el) => {
                  if (!el.points || el.points.length < 3) return null;
                  const path =
                    el.points
                      .map(
                        (p, i) =>
                          `${i === 0 ? "M" : "L"} ${p.x * displayedScale} ${
                            p.y * displayedScale
                          }`
                      )
                      .join(" ") + " Z";
                  return (
                    <path
                      key={el.id}
                      d={path}
                      fill="rgba(0,0,0,0.04)"
                      stroke="none"
                    />
                  );
                })}

              {/* 선 */}
              {elements
                .filter(
                  (el) =>
                    el.type === "건물윤곽" &&
                    (el.floor ?? 0) === currentFloorIndex
                )
                .map((el) => {
                  const pts = el.points || [];
                  return pts.map((p, idx) => {
                    const a = p;
                    const b = pts[(idx + 1) % pts.length];
                    const isHoverSeg =
                      hover &&
                      hover.kind === "contour-seg" &&
                      hover.id === el.id &&
                      hover.segIndex === idx;

                    const strokeW = isHoverSeg ? 4 : 2;
                    const strokeColor = isHoverSeg
                      ? "rgba(0,120,220,0.95)"
                      : "rgba(0,0,0,0.6)";

                    return (
                      <line
                        key={`${el.id}-seg-${idx}`}
                        x1={a.x * displayedScale}
                        y1={a.y * displayedScale}
                        x2={b.x * displayedScale}
                        y2={b.y * displayedScale}
                        stroke={strokeColor}
                        strokeWidth={strokeW}
                        pointerEvents="none"
                      />
                    );
                  });
                })}

              {/* 윤곽 프리뷰 */}
              {outlinePoints.length > 0 && (
                <>
                  <polyline
                    points={outlinePoints
                      .map(
                        (p) => `${p.x * displayedScale},${p.y * displayedScale}`
                      )
                      .join(" ")}
                    fill="none"
                    stroke="rgba(0,120,220,0.9)"
                    strokeWidth={2}
                    strokeDasharray="6"
                  />
                  {outlinePoints.map((p, idx) => (
                    <circle
                      key={idx}
                      cx={p.x * displayedScale}
                      cy={p.y * displayedScale}
                      r={4}
                      fill="rgba(0,120,220,0.9)"
                    />
                  ))}
                </>
              )}
            </svg>

            {elements
              .filter((el) => (el.floor ?? 0) === currentFloorIndex)
              .map((el) => {
                // =========================
                // 1) 문
                // =========================
                if (el.type === "문") {
                  const dispX = natToDispX(el.x);
                  const dispY = natToDispY(el.y);
                  const doorW = 28 * displayedScale;
                  const doorH = 10 * displayedScale;

                  const transform = `translate(${dispX - doorW / 2}px, ${
                    dispY - doorH / 2
                  }px) rotate(${el.angle || 0}deg)`;

                  const isRotating = rotatingDoorId === el.id;
                  const isHoverDoor =
                    hover && hover.kind === "door" && hover.id === el.id;
                  const isSelectedDoor = selectedIds.includes(el.id);

                  return (
                    <div
                      key={el.id}
                      style={{
                        position: "absolute",
                        left: 0,
                        top: 0,
                        width: VIEW_W,
                        height: VIEW_H,
                        pointerEvents: "none",
                        zIndex: isSelectedDoor ? 50 : 35,
                      }}
                    >
                      <div
                        onMouseDown={(ev) => {
                          if (ev.button === 2) return;
                          if (mode === null) {
                            ev.stopPropagation();
                            const nat = clientToNatural(ev.clientX, ev.clientY);

                            // 선택 처리
                            const elId = el.id;
                            const current = selectedIds.length
                              ? selectedIds
                              : selectedIdsRef.current;

                            let newSelectedIds;
                            if (ev.ctrlKey) {
                              newSelectedIds = current.includes(elId)
                                ? current.filter((id) => id !== elId)
                                : [...current, elId];
                            } else {
                              newSelectedIds = [elId];
                            }

                            setSelectedIds(newSelectedIds);
                            setSelectedId(
                              newSelectedIds[newSelectedIds.length - 1]
                            );
                            setContextMenu(null);
                            setEditingResizeId(null);

                            // 드래그 시작
                            pushUndoSnapshot();
                            startElementsDrag(newSelectedIds, nat);
                          }
                        }}
                        style={{
                          position: "absolute",
                          transformOrigin: "center",
                          transform,
                          left: 0,
                          top: 0,
                          width: doorW,
                          height: doorH,
                          pointerEvents: "auto",
                          cursor:
                            mode === null
                              ? "grab"
                              : mode === "문"
                              ? "crosshair"
                              : "default",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <div
                          style={{
                            width: doorW,
                            height: doorH,
                            background: "rgba(60,60,60,0.95)",
                            border: isRotating
                              ? "3px solid #ffb84d"
                              : isSelectedDoor
                              ? "2px solid #0b74de"
                              : isHoverDoor
                              ? "2px dashed #0b74de"
                              : "1px solid rgba(0,0,0,0.4)",
                            borderRadius: 2,
                            boxSizing: "border-box",
                          }}
                        />
                      </div>
                    </div>
                  );
                }

                // =========================
                // 2) 비상구
                // =========================
                if (el.type === "비상구") {
                  const dispX = natToDispX(el.x);
                  const dispY = natToDispY(el.y);
                  const sizeW = (el.width || 40) * displayedScale;
                  const sizeH = (el.height || 40) * displayedScale;

                  const isSelectedExit = selectedIds.includes(el.id);
                  const isHoverExit =
                    hover && hover.kind === "exit" && hover.id === el.id;

                  return (
                    <div
                      key={el.id}
                      onMouseDown={(ev) => {
                        if (ev.button === 2) return;
                        if (mode === null) {
                          ev.stopPropagation();
                          const nat = clientToNatural(ev.clientX, ev.clientY);
                          pushUndoSnapshot();

                          const elId = el.id;
                          const current = selectedIds.length
                            ? selectedIds
                            : selectedIdsRef.current;

                          let newSelectedIds;
                          if (ev.ctrlKey) {
                            newSelectedIds = current.includes(elId)
                              ? current.filter((id) => id !== elId)
                              : [...current, elId];
                          } else {
                            newSelectedIds = [elId];
                          }

                          setSelectedIds(newSelectedIds);
                          setSelectedId(
                            newSelectedIds[newSelectedIds.length - 1]
                          );
                          setContextMenu(null);
                          setEditingResizeId(null);
                          startElementsDrag(newSelectedIds, nat);
                        }
                      }}
                      style={{
                        position: "absolute",
                        left: dispX - sizeW / 2,
                        top: dispY - sizeH / 2,
                        width: sizeW,
                        height: sizeH,
                        backgroundColor: "#0f9d58",
                        border: isSelectedExit
                          ? "3px solid #0b74de"
                          : isHoverExit
                          ? "2px dashed #ff8800"
                          : "2px solid #0b5c2f",
                        borderRadius: 4,
                        boxSizing: "border-box",
                        zIndex: isSelectedExit ? 42 : 32,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "white",
                        fontSize: 10,
                        fontWeight: "bold",
                        pointerEvents: "auto",
                      }}
                    >
                      EXIT
                    </div>
                  );
                }

                // =========================
                // 3) 건물윤곽은 여기서 렌더 X (SVG에서 렌더)
                // =========================
                if (el.type === "건물윤곽") return null;

                // =========================
                // 4) 박스(방/제한/재난/안전)
                // =========================
                const isHoverBox =
                  hover && hover.kind === "box" && hover.id === el.id;
                const isSelectedBox = selectedIds.includes(el.id);
                const boxW = Math.max(1, natToDispW(el.width));
                const boxH = Math.max(1, natToDispH(el.height));

                return (
                  <div
                    key={el.id}
                    onMouseDown={(ev) => {
                      if (ev.button === 2) return;
                      if (mode === null) {
                        ev.stopPropagation();
                        const nat = clientToNatural(ev.clientX, ev.clientY);
                        pushUndoSnapshot();

                        const elId = el.id;
                        const current = selectedIds.length
                          ? selectedIds
                          : selectedIdsRef.current;

                        let newSelectedIds;
                        if (ev.ctrlKey) {
                          newSelectedIds = current.includes(elId)
                            ? current.filter((id) => id !== elId)
                            : [...current, elId];
                        } else {
                          newSelectedIds = [elId];
                        }

                        setSelectedIds(newSelectedIds);
                        setSelectedId(
                          newSelectedIds[newSelectedIds.length - 1]
                        );
                        setContextMenu(null);
                        setEditingResizeId(null);
                        startElementsDrag(newSelectedIds, nat);
                      }
                    }}
                    style={{
                      position: "absolute",
                      left: natToDispX(el.x),
                      top: natToDispY(el.y),
                      width: boxW,
                      height: boxH,
                      backgroundColor: typeColor(el.type, 0.25),
                      border: isSelectedBox
                        ? "3px solid #0b74de"
                        : isHoverBox
                        ? "2px dashed #ff8800"
                        : "2px solid rgba(0,0,0,0.5)",
                      boxSizing: "border-box",
                      zIndex: isSelectedBox ? 40 : 30,
                      pointerEvents: "auto",
                      display: "flex",
                      alignItems: "flex-start",
                    }}
                  >
                    {el.name && (
                      <div
                        style={{
                          background: "rgba(255,255,255,0.9)",
                          padding: "2px 6px",
                          fontSize: 12,
                          borderRadius: 4,
                          margin: 4,
                        }}
                      >
                        {el.name}
                      </div>
                    )}

                    {renderResizeHandles(el)}
                  </div>
                );
              })}

            {/* 박스 생성 미리보기 */}
            {preview && (
              <div
                style={{
                  position: "absolute",
                  left: natToDispX(preview.x),
                  top: natToDispY(preview.y),
                  width: Math.max(1, natToDispW(preview.width)),
                  height: Math.max(1, natToDispH(preview.height)),
                  border: "2px dashed #ff6600",
                  backgroundColor: "rgba(255, 165, 0, 0.12)",
                  zIndex: 60,
                  pointerEvents: "none",
                }}
              />
            )}

            {/* 드래그 박스(박스 선택) */}
            {selectionRect && (
              <div
                style={{
                  position: "absolute",
                  left: selectionRect.x,
                  top: selectionRect.y,
                  width: selectionRect.w,
                  height: selectionRect.h,
                  border: "1px dashed #0b74de",
                  background: "rgba(11, 116, 222, 0.08)",
                  zIndex: 90,
                  pointerEvents: "none",
                }}
              />
            )}
          </div>

          {/* 오른쪽 상태/안내 패널 */}
          <div className="w-[320px] shrink-0">
            <div className="bg-white border shadow rounded p-4 space-y-2 sticky top-32">
              <div className="text-sm text-gray-700">
                <b>현재 모드:</b> {mode ?? "선택"}
              </div>
              <div className="text-sm text-gray-700">
                <b>선택:</b>{" "}
                {selectedIds.length ? `${selectedIds.length}개` : "없음"}
              </div>
              <div className="text-sm text-gray-700">
                <b>팁:</b> Space+드래그로 패닝, Ctrl+휠로 줌
              </div>

              <div className="pt-2 border-t text-xs text-gray-500 space-y-1">
                <div>• 우클릭: 삭제/타입변경/리사이즈/문 회전</div>
                <div>• ESC: 모드/드래그/리사이즈/회전 취소</div>
              </div>
            </div>
          </div>
        </div>

        {/* 컨텍스트 메뉴 */}
        {contextMenu && (
          <div
            ref={menuRef}
            style={{
              position: "fixed",
              left: contextMenu.x,
              top: contextMenu.y,
              transform: contextMenu.flipY
                ? "translateY(-100%)"
                : "translateY(0)",
              background: "#fff",
              border: "1px solid #ccc",
              borderRadius: 6,
              boxShadow: "0 6px 18px rgba(0,0,0,0.12)",
              zIndex: 200,
              minWidth: 220,
              overflow: "hidden",
            }}
          >
            {contextMenu.options.map((opt, i) => (
              <div
                key={i}
                onClick={() => opt.action()}
                style={{
                  padding: "10px 12px",
                  cursor: "pointer",
                  borderBottom:
                    i === contextMenu.options.length - 1
                      ? "none"
                      : "1px solid #eee",
                  fontSize: 13,
                }}
                onMouseEnter={(ev) =>
                  (ev.currentTarget.style.background = "#f5f5f5")
                }
                onMouseLeave={(ev) =>
                  (ev.currentTarget.style.background = "#fff")
                }
              >
                {opt.label}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
