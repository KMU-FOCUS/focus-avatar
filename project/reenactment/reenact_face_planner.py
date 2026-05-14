from __future__ import annotations

# raw metadata의 face 목록을 reenact용 프레임 계획으로 바꿀 때 쓰는 가벼운 헬퍼 모음이다.
# 어떤 얼굴을 고를지, 어떤 bbox/landmark 좌표계를 쓸지를 결정하는 역할을 한다.

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .reenact_assets_runtime import CROP_SIZE


def bbox_area(face: Mapping[str, Any]) -> float:
    # identity 기준이 애매할 때는 bbox 크기를 가장 단순한 fallback 기준으로 쓴다.
    # 즉 한 프레임 안에서 "어떤 얼굴을 고를지"를 면적으로 결정한다.
    bbox = face.get("bbox")
    if not isinstance(bbox, Mapping):
        return -1.0
    return float(bbox.get("width", 0.0)) * float(bbox.get("height", 0.0))


def dominant_tracking_id(frames: list[Mapping[str, Any]]) -> int | None:
    # dominant_track은 클립 전체에서 가장 자주 등장한 tracking_id를 뜻한다.
    counts: Counter[int] = Counter()
    for frame in frames:
        for face in frame.get("faces", []) or []:
            tracking_id = face.get("tracking_id")
            if tracking_id is not None:
                counts[int(tracking_id)] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def pick_face(
    faces: list[Mapping[str, Any]],
    *,
    face_mode: str,
    explicit_tracking_id: int | None,
    dominant_track_id: int | None,
) -> Mapping[str, Any] | None:
    # 우선순위는 다음과 같다.
    # 1) 사용자가 명시한 tracking_id
    # 2) dominant_track 모드일 때의 대표 id
    # 3) 그 외에는 현재 프레임에서 가장 큰 bbox
    if not faces:
        return None

    if explicit_tracking_id is not None:
        matched = [face for face in faces if int(face.get("tracking_id", -1)) == explicit_tracking_id]
        if matched:
            return max(matched, key=bbox_area)

    if face_mode == "dominant_track" and dominant_track_id is not None:
        matched = [face for face in faces if int(face.get("tracking_id", -1)) == dominant_track_id]
        if matched:
            return max(matched, key=bbox_area)

    return max(faces, key=bbox_area)


def select_faces(
    faces: list[Mapping[str, Any]],
    *,
    face_mode: str,
    explicit_tracking_id: int | None,
    dominant_track_id: int | None,
    process_all_faces: bool,
) -> list[Mapping[str, Any]]:
    # 단일 얼굴 모드에서는 선택된 얼굴 하나만 돌려준다.
    # process_all_faces 모드에서는 남은 얼굴들을 bbox 큰 순서대로 모두 돌려준다.
    if not faces:
        return []

    if explicit_tracking_id is not None:
        matched = [face for face in faces if int(face.get("tracking_id", -1)) == explicit_tracking_id]
        return sorted(matched, key=bbox_area, reverse=True)

    if process_all_faces:
        return sorted(faces, key=bbox_area, reverse=True)

    picked = pick_face(
        faces,
        face_mode=face_mode,
        explicit_tracking_id=None,
        dominant_track_id=dominant_track_id,
    )
    return [picked] if picked is not None else []


def filter_faces(
    faces: list[Mapping[str, Any]],
    *,
    excluded_tracking_ids: set[int],
) -> list[Mapping[str, Any]]:
    # 선택 로직과 제외 로직을 분리해 두면
    # 나중에 process_all_faces 와 사용자 skip 목록을 더 깔끔하게 조합할 수 있다.
    if not excluded_tracking_ids:
        return faces
    filtered: list[Mapping[str, Any]] = []
    for face in faces:
        tracking_id = face.get("tracking_id")
        if tracking_id is not None and int(tracking_id) in excluded_tracking_ids:
            continue
        filtered.append(face)
    return filtered


def image_landmarks_to_crop_points(
    landmarks_image_xy: np.ndarray,
    bbox_xyxy: Sequence[float],
    *,
    crop_size: int = CROP_SIZE,
) -> np.ndarray:
    # 이미지 좌표계의 landmark를 bbox 내부 기준 crop 좌표계로 변환한다.
    # warp_face()는 정규화된 crop 좌표계를 기대하므로 이 변환이 필요하다.
    # 즉 영상 공간에서 맞춘 landmark를 warp용 crop 공간으로 넘기는 다리 역할이다.
    x1, y1, x2, y2 = map(float, bbox_xyxy[:4])
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    points = np.asarray(landmarks_image_xy, dtype=np.float32).reshape(-1, 2).copy()
    points[:, 0] = (points[:, 0] - x1) / width * crop_size
    points[:, 1] = (points[:, 1] - y1) / height * crop_size
    return points
