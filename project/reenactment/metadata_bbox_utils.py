from __future__ import annotations

# Qualcomm metadata 기반 reenact 파이프라인에서 공통으로 쓰는 bbox/landmark helper 모듈이다.
# 이 파일은 직접 실행하는 CLI 스크립트가 아니라, 다른 파이프라인 파일에서 import해서 사용하는 용도다.
# 주요 역할은 다음과 같다.
# 1. metadata JSON을 읽는다.
# 2. detector bbox(x, y, width, height)를 xyxy 형식으로 바꾼다.
# 3. bbox를 프레임 경계에 맞게 자르고, scale/shift/smoothing 보정을 적용한다.
# 4. Qualcomm 3DMM coeff에서 필요한 264차원 coefficient를 꺼낸다.
# 5. 복원된 landmark를 detection bbox 내부에 맞게 재배치한다.
# 6. 필요할 때 디버그용 bbox/landmark overlay를 그린다.

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


def load_json(path: str | Path) -> Any:
    # metadata_*.json 파일을 읽어서 Python dict/list 구조로 변환한다.
    # pipeline 쪽에서는 이 함수로 frames 배열을 읽은 뒤, frame index별 face 정보를 사용한다.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bbox_to_xyxy(bbox: Mapping[str, Any] | None) -> tuple[int, int, int, int] | None:
    # metadata에 저장된 detector bbox는 보통 x, y, width, height 형식이다.
    # 하지만 scale/shift/clamp/composite 단계에서는 x1, y1, x2, y2 형식이 더 다루기 쉽다.
    # 그래서 여기서 모든 bbox를 xyxy 형식으로 통일한다.
    # bbox가 없거나 width/height가 0 이하이면 이후 합성이 불가능하므로 None을 반환한다.
    if not isinstance(bbox, Mapping):
        return None
    x = int(round(float(bbox.get("x", 0.0))))
    y = int(round(float(bbox.get("y", 0.0))))
    width = int(round(float(bbox.get("width", 0.0))))
    height = int(round(float(bbox.get("height", 0.0))))
    if width <= 0 or height <= 0:
        return None
    return x, y, x + width, y + height


def clamp_box(
    box: tuple[int, int, int, int] | tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int] | None:
    # bbox가 영상 프레임 밖으로 나가는 경우를 막기 위해 좌표를 프레임 범위 안으로 자른다.
    # 예를 들어 얼굴이 화면 가장자리에 있으면 x1/y1/x2/y2 일부가 음수거나 프레임 크기를 넘을 수 있다.
    # clipping 후에도 폭이나 높이가 0 이하라면 실제로 합성할 영역이 없으므로 None을 반환한다.
    x1, y1, x2, y2 = box
    x1 = int(round(max(0, min(frame_w - 1, float(x1)))))
    y1 = int(round(max(0, min(frame_h - 1, float(y1)))))
    x2 = int(round(max(0, min(frame_w - 1, float(x2)))))
    y2 = int(round(max(0, min(frame_h - 1, float(y2)))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def scale_box(
    box: tuple[int, int, int, int],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    # bbox의 중심점은 유지한 채 폭/높이만 scale한다.
    # 중심 기준으로 키워야 얼굴 위치가 좌상단 기준으로 밀리는 drift가 생기지 않는다.
    # scale_x/scale_y를 따로 두면 가로/세로를 독립적으로 넓히거나 줄일 수 있다.
    # reenact에서는 합성 crop이 너무 타이트하거나 넓을 때 이 단계로 ROI 크기를 보정한다.
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    width = max(1.0, (x2 - x1) * float(scale_x))
    height = max(1.0, (y2 - y1) * float(scale_y))
    half_w = 0.5 * width
    half_h = 0.5 * height
    return (
        int(round(cx - half_w)),
        int(round(cy - half_h)),
        int(round(cx + half_w)),
        int(round(cy + half_h)),
    )


def collect_metadata_bbox_extent(frames: list[Any]) -> tuple[int, int]:
    max_x2 = 0.0
    max_y2 = 0.0
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        for face in frame.get("faces") or []:
            if not isinstance(face, Mapping):
                continue
            bbox = face.get("bbox")
            if not isinstance(bbox, Mapping):
                continue
            try:
                x = float(bbox.get("x", 0.0))
                y = float(bbox.get("y", 0.0))
                w = float(bbox.get("width", 0.0))
                h = float(bbox.get("height", 0.0))
            except (TypeError, ValueError):
                continue
            max_x2 = max(max_x2, x + w)
            max_y2 = max(max_y2, y + h)
    return int(round(max_x2)), int(round(max_y2))


def infer_metadata_canvas_size(
    frames: list[Any],
    *,
    video_w: int,
    video_h: int,
    explicit_width: int | None,
    explicit_height: int | None,
) -> tuple[int, int]:
    if explicit_width is not None and explicit_height is not None:
        return max(1, int(explicit_width)), max(1, int(explicit_height))

    extent_w, extent_h = collect_metadata_bbox_extent(frames)
    if explicit_width is not None:
        return max(1, int(explicit_width)), max(1, extent_h, video_h)
    if explicit_height is not None:
        return max(1, extent_w, video_w), max(1, int(explicit_height))
    if extent_w <= 0 or extent_h <= 0:
        return max(1, int(video_w)), max(1, int(video_h))

    candidate_sizes = [
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
        (1280, 720),
    ]
    video_aspect = float(video_w) / float(max(video_h, 1))
    for candidate_w, candidate_h in candidate_sizes:
        if extent_w <= candidate_w and extent_h <= candidate_h:
            candidate_aspect = float(candidate_w) / float(candidate_h)
            if abs(candidate_aspect - video_aspect) < 0.02:
                return candidate_w, candidate_h

    return max(extent_w, int(video_w)), max(extent_h, int(video_h))


def remap_box_to_canvas(
    box: tuple[int, int, int, int],
    *,
    source_w: int,
    source_h: int,
    target_w: int,
    target_h: int,
) -> tuple[int, int, int, int]:
    scale_x = float(target_w) / float(max(source_w, 1))
    scale_y = float(target_h) / float(max(source_h, 1))
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )


def extract_coeff_264(face: Mapping[str, Any]) -> np.ndarray | None:
    # Qualcomm FaceMap 3DMM metadata에서 tdmm_raw.coeffs를 꺼낸다.
    # downstream 함수들은 앞쪽 264개 coeff를 기준으로 landmark 복원/pose 계산을 수행한다.
    # coeff가 없거나 길이가 부족하면 해당 얼굴은 reenact에 필요한 정보가 부족하므로 None을 반환한다.
    # 반환 시 copy()를 해서 원본 metadata 배열과 분리된 float32 벡터로 사용한다.
    tdmm = face.get("tdmm_raw")
    if not isinstance(tdmm, Mapping):
        return None
    coeffs = tdmm.get("coeffs")
    if coeffs is None:
        return None
    vector = np.asarray(coeffs, dtype=np.float32).reshape(-1)
    if vector.size < 264:
        return None
    return vector[:264].copy()


def shift_box(
    box: tuple[int, int, int, int],
    *,
    shift_x: int,
    shift_y: int,
) -> tuple[int, int, int, int]:
    # bbox 전체를 x/y 방향으로 같은 픽셀만큼 이동한다.
    # metadata bbox와 실제 합성 위치가 미세하게 어긋날 때 수동 보정용으로 사용한다.
    # shift_x가 양수면 오른쪽, 음수면 왼쪽으로 이동한다.
    # shift_y가 양수면 아래쪽, 음수면 위쪽으로 이동한다.
    x1, y1, x2, y2 = box
    return x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y


def smooth_box(
    box: tuple[int, int, int, int],
    *,
    tracking_id: int | None,
    state: dict[int, np.ndarray],
    smooth_factor: float,
) -> tuple[int, int, int, int]:
    # tracking_id별로 이전 bbox를 기억해 두고, 갑자기 bbox가 튀는 경우에만 EMA smoothing을 적용한다.
    # 평소에는 metadata bbox를 그대로 사용해서 반응성을 유지한다.
    # center/size 변화량이 threshold보다 작으면 정상 움직임으로 보고 smoothing하지 않는다.
    # 변화량이 크면 detection jitter일 가능성이 있으므로 previous와 current를 섞어서 안정화한다.
    # tracking_id가 없으면 같은 얼굴을 이어서 추적할 수 없으므로 smoothing하지 않고 원본 box를 반환한다.
    if tracking_id is None or smooth_factor <= 0.0:
        return box

    current = np.asarray(box, dtype=np.float32)
    previous = state.get(int(tracking_id))
    if previous is None:
        state[int(tracking_id)] = current
        return box

    prev_cx = 0.5 * (previous[0] + previous[2])
    prev_cy = 0.5 * (previous[1] + previous[3])
    curr_cx = 0.5 * (current[0] + current[2])
    curr_cy = 0.5 * (current[1] + current[3])
    prev_w = max(previous[2] - previous[0], 1.0)
    prev_h = max(previous[3] - previous[1], 1.0)
    curr_w = max(current[2] - current[0], 1.0)
    curr_h = max(current[3] - current[1], 1.0)

    center_jump = float(np.hypot(curr_cx - prev_cx, curr_cy - prev_cy))
    center_jump_threshold = max(12.0, 0.08 * max(prev_w, prev_h))
    width_ratio_change = abs(curr_w - prev_w) / prev_w
    height_ratio_change = abs(curr_h - prev_h) / prev_h
    size_jump_threshold = 0.12

    if (
        center_jump < center_jump_threshold
        and width_ratio_change < size_jump_threshold
        and height_ratio_change < size_jump_threshold
    ):
        state[int(tracking_id)] = current
        return box

    smoothed = previous * smooth_factor + current * (1.0 - smooth_factor)
    state[int(tracking_id)] = smoothed
    return tuple(int(round(v)) for v in smoothed.tolist())


def fit_landmarks_to_bbox(
    landmarks_crop_xy: np.ndarray,
    det_bbox_xyxy: tuple[int, int, int, int],
    *,
    pad_left: float,
    pad_right: float,
    pad_top: float,
    pad_bottom: float,
) -> np.ndarray:
    # Qualcomm coeff로 복원한 landmark는 자체 crop 좌표계에 가까운 형태라서,
    # 실제 영상 속 detection bbox 위치에 맞게 다시 scale/translate 해야 한다.
    # 이 함수는 landmark의 원래 bounding range를 구한 뒤, detection bbox 내부의 padded 영역으로 선형 매핑한다.
    # pad_left/right/top/bottom은 bbox 안에서 landmark가 차지할 내부 여백 비율이다.
    # 예를 들어 pad_right를 키우면 landmark가 오른쪽 경계에서 더 떨어져 배치된다.
    pts = np.asarray(landmarks_crop_xy, dtype=np.float32).reshape(-1, 2).copy()

    src_x1 = float(np.min(pts[:, 0]))
    src_y1 = float(np.min(pts[:, 1]))
    src_x2 = float(np.max(pts[:, 0]))
    src_y2 = float(np.max(pts[:, 1]))
    src_w = max(src_x2 - src_x1, 1e-6)
    src_h = max(src_y2 - src_y1, 1e-6)

    x1, y1, x2, y2 = map(float, det_bbox_xyxy)
    bw = max(x2 - x1, 1e-6)
    bh = max(y2 - y1, 1e-6)

    dst_x1 = x1 + bw * pad_left
    dst_x2 = x2 - bw * pad_right
    dst_y1 = y1 + bh * pad_top
    dst_y2 = y2 - bh * pad_bottom

    dst_w = max(dst_x2 - dst_x1, 1e-6)
    dst_h = max(dst_y2 - dst_y1, 1e-6)

    pts[:, 0] = (pts[:, 0] - src_x1) / src_w * dst_w + dst_x1
    pts[:, 1] = (pts[:, 1] - src_y1) / src_h * dst_h + dst_y1
    return pts


def draw_landmarks(frame: Any, landmarks_xy: np.ndarray, *, radius: int, color: tuple[int, int, int]) -> None:
    # 디버깅용으로 landmark 점들을 영상 프레임 위에 그린다.
    # 실제 reenact 합성에는 필수는 아니지만, coeff 복원 결과가 bbox 안에 잘 맞는지 확인할 때 유용하다.
    # 프레임 밖으로 나간 점은 그리지 않는다.
    frame_h, frame_w = frame.shape[:2]
    for point in np.asarray(landmarks_xy, dtype=np.float32):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if x < 0 or x >= frame_w or y < 0 or y >= frame_h:
            continue
        cv2.circle(frame, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)


def draw_label(frame: Any, text: str, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    # bbox 왼쪽 위에 tracking_id나 face label을 읽기 쉬운 배경 박스와 함께 그린다.
    # 디버그 overlay에서 어떤 얼굴 track이 어떤 bbox와 연결되는지 확인하기 위한 용도다.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y1 - text_h - baseline - 8)
    bottom = max(text_h + baseline + 8, y1)
    right = min(frame.shape[1] - 1, x1 + text_w + 12)
    cv2.rectangle(frame, (x1, top), (right, bottom), color, thickness=-1)
    cv2.putText(
        frame,
        text,
        (x1 + 6, bottom - baseline - 4),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_bbox_overlay(
    frame: Any,
    bbox_xyxy: tuple[int, int, int, int],
    *,
    tracking_id: int | None,
    line_thickness: int,
    hide_labels: bool,
    color: tuple[int, int, int] = (80, 220, 120),
) -> None:
    # 최종 composite ROI를 눈으로 확인할 때 쓰는 디버그용 bbox 시각화 함수다.
    # 입력 bbox는 다시 한 번 clamp해서 프레임 밖으로 나가는 rectangle draw 오류를 막는다.
    # hide_labels=False이면 tracking_id label도 함께 표시한다.
    frame_h, frame_w = frame.shape[:2]
    clamped = clamp_box(bbox_xyxy, frame_w, frame_h)
    if clamped is None:
        return
    x1, y1, x2, y2 = clamped
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=line_thickness)
    if not hide_labels:
        label = f"id {tracking_id}" if tracking_id is not None else "face"
        draw_label(frame, label, x1, y1, color)
