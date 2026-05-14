from __future__ import annotations

import bisect
import random
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Mapping

import numpy as np

from shared.converters.coeffs_to_landmark import reconstruct_qualcomm_68_landmarks


from . import metadata_bbox_utils as overlay_helpers
from .reenact_composite import warp_face
from .reenact_face_planner import (
    bbox_area,
    filter_faces,
    image_landmarks_to_crop_points,
)
from .reenact_restore import restore_keyframe_face_region
from .reenact_assets_runtime import (
    coeff_to_pose_radians,
    load_avatar_view_assets,
    select_avatar_view,
)

# 이 파일은 Qualcomm 3DMM metadata를 기반으로 avatar 얼굴 합성용 keyframe을 미리 계산하는 모듈이다.
# 전체 흐름은 다음과 같다.
# 1. metadata frame마다 처리할 얼굴을 고르고, bbox/coeff/avatar/source_view 정보를 FrameFacePlan으로 저장한다.
# 2. 모든 프레임을 직접 warp하지 않고, 일정 간격의 keyframe만 실제 warp_face로 계산한다.
# 3. 렌더링 단계에서는 미리 만든 keyframe cache를 재사용하고, 필요한 경우 이전 keyframe과 현재 keyframe을 보간한다.
# 이렇게 나누면 무거운 face warp 계산을 줄이면서도 프레임별 얼굴 위치와 avatar 선택을 안정적으로 유지할 수 있다.

DEFAULT_BBOX_SCALE_X = 1.06
DEFAULT_BBOX_SCALE_Y = 1.06
DEFAULT_BBOX_SHIFT_X = 4
DEFAULT_BBOX_SHIFT_Y = 0
DEFAULT_BBOX_SMOOTH_FACTOR = 0.65

DEFAULT_PAD_LEFT = 0.08
DEFAULT_PAD_RIGHT = 0.06
DEFAULT_PAD_TOP = 0.16
DEFAULT_PAD_BOTTOM = 0.04


@dataclass(frozen=True)
class FrameFacePlan:
    # 한 프레임에서 한 얼굴을 어떻게 처리할지 기록한 "계획표" 한 줄이다.
    # 여기에는 실제 합성된 얼굴 이미지가 들어가지 않는다.
    # 대신 keyframe 계산에 필요한 원본 정보만 저장한다.
    # - frame_index: 원본 비디오 기준 프레임 번호
    # - face_key: 같은 사람/같은 slot을 계속 추적하기 위한 내부 key
    # - tracking_id: metadata에 tracking_id가 있을 때의 실제 추적 번호
    # - bbox_xyxy: 이 얼굴을 합성할 위치
    # - coeff_264: Qualcomm 3DMM 계수
    # - avatar_id/source_view: 어떤 avatar의 어떤 방향 이미지를 쓸지에 대한 선택 결과
    frame_index: int
    face_key: str
    tracking_id: int | None
    bbox_xyxy: tuple[float, float, float, float]
    coeff_264: np.ndarray
    avatar_id: str
    source_view: str


@dataclass(frozen=True)
class WarpedKeyframe:
    # 실제 keyframe 계산 결과다.
    # FrameFacePlan이 "계획"이라면, WarpedKeyframe은 warp_face까지 끝난 "결과물"이다.
    # 이후 렌더링 루프는 이 결과를 다시 사용하므로, 무거운 warp 계산을 매 프레임 반복하지 않는다.
    # - face_bgr: avatar 얼굴을 target crop 형태로 변형한 이미지
    # - mask_uint8: 합성할 얼굴 영역 mask
    # - crop_points: 현재 프레임 bbox 내부의 target landmark 위치
    # - avatar_id/source_view: 이 keyframe을 만들 때 사용한 avatar 정보
    frame_index: int
    face_bgr: np.ndarray
    mask_uint8: np.ndarray
    crop_points: np.ndarray
    avatar_id: str
    source_view: str


def _bbox_center(box_xyxy: tuple[int, int, int, int] | tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    return np.asarray([(float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5], dtype=np.float32)


def _allocate_untracked_face_key(
    det_bbox_xyxy: tuple[int, int, int, int] | tuple[float, float, float, float],
    *,
    slot_index: int,
    untracked_face_centers: dict[str, np.ndarray],
    used_untracked_face_keys: set[str],
) -> str:
    # tracking_id가 없는 얼굴은 프레임 내 순서 대신 최근 bbox 중심 위치에 매칭한다.
    # 그래야 bbox 면적 순서가 바뀌어도 같은 얼굴이 기존 avatar/keyframe cache를 계속 재사용할 수 있다.
    center = _bbox_center(det_bbox_xyxy)
    width = max(float(det_bbox_xyxy[2]) - float(det_bbox_xyxy[0]), 1.0)
    height = max(float(det_bbox_xyxy[3]) - float(det_bbox_xyxy[1]), 1.0)
    reuse_distance_threshold = max(width, height) * 0.75

    best_key: str | None = None
    best_distance: float | None = None
    for face_key, previous_center in untracked_face_centers.items():
        if face_key in used_untracked_face_keys:
            continue
        distance = float(np.linalg.norm(center - previous_center))
        if best_distance is None or distance < best_distance:
            best_key = face_key
            best_distance = distance

    if best_key is not None and best_distance is not None and best_distance <= reuse_distance_threshold:
        untracked_face_centers[best_key] = center
        used_untracked_face_keys.add(best_key)
        return best_key

    suffix = len(untracked_face_centers)
    candidate = f"slot:{slot_index}"
    while candidate in untracked_face_centers or candidate in used_untracked_face_keys:
        suffix += 1
        candidate = f"untracked:{suffix}"
    untracked_face_centers[candidate] = center
    used_untracked_face_keys.add(candidate)
    return candidate


def choose_avatar_id_for_face(
    *,
    face_key: str,
    avatar_ids: list[str],
    assignment_by_face: dict[str, str],
    rng: random.Random,
) -> str:
    # 같은 face_key에는 항상 같은 avatar를 배정한다.
    # 이미 배정된 avatar가 있으면 그대로 재사용해서 프레임마다 얼굴이 바뀌는 문제를 막는다.
    # 아직 배정되지 않은 얼굴이면, 가능한 한 다른 얼굴과 겹치지 않는 avatar를 먼저 고른다.
    existing = assignment_by_face.get(face_key)
    if existing is not None:
        return existing

    used_avatar_ids = set(assignment_by_face.values())
    available_avatar_ids = [avatar_id for avatar_id in avatar_ids if avatar_id not in used_avatar_ids]
    if not available_avatar_ids:
        available_avatar_ids = list(avatar_ids)
    chosen_avatar_id = rng.choice(available_avatar_ids)
    assignment_by_face[face_key] = chosen_avatar_id
    return chosen_avatar_id


def build_single_face_plan(
    *,
    frame_index: int,
    selected: Mapping[str, Any],
    slot_index: int,
    smooth_state: dict[int, np.ndarray],
    untracked_face_centers: dict[str, np.ndarray],
    used_untracked_face_keys: set[str],
    avatar_ids: list[str],
    load_avatar_profile_for_id: Callable[[str], dict[str, Any]],
    avatar_assignment_by_face: dict[str, str],
    avatar_rng: random.Random,
) -> FrameFacePlan | None:
    # metadata의 얼굴 하나를 FrameFacePlan 하나로 변환한다.
    # coeff나 bbox가 없으면 이후 landmark 복원/합성이 불가능하므로 None을 반환한다.
    coeff_264 = overlay_helpers.extract_coeff_264(selected)
    det_bbox_xyxy = overlay_helpers.bbox_to_xyxy(selected.get("bbox"))
    if coeff_264 is None or det_bbox_xyxy is None:
        return None

    tracking_id = selected.get("tracking_id")
    tracking_id_int = int(tracking_id) if tracking_id is not None else None

    # bbox 크기/위치/smoothing 보정 단계다.
    # crop이 너무 작으면 얼굴 일부가 잘리고, 너무 크면 배경까지 많이 합성될 수 있다.
    # smoothing은 프레임마다 bbox가 미세하게 떨리는 현상을 줄이는 역할을 한다.
    det_bbox_xyxy = overlay_helpers.scale_box(
        det_bbox_xyxy,
        scale_x=DEFAULT_BBOX_SCALE_X,
        scale_y=DEFAULT_BBOX_SCALE_Y,
    )
    det_bbox_xyxy = overlay_helpers.shift_box(
        det_bbox_xyxy,
        shift_x=DEFAULT_BBOX_SHIFT_X,
        shift_y=DEFAULT_BBOX_SHIFT_Y,
    )
    det_bbox_xyxy = overlay_helpers.smooth_box(
        det_bbox_xyxy,
        tracking_id=tracking_id_int,
        state=smooth_state,
        smooth_factor=DEFAULT_BBOX_SMOOTH_FACTOR,
    )

    # face_key를 기준으로 같은 얼굴에 같은 avatar를 계속 붙인다.
    if tracking_id_int is not None:
        face_key = f"track:{tracking_id_int}"
    else:
        face_key = _allocate_untracked_face_key(
            det_bbox_xyxy,
            slot_index=slot_index,
            untracked_face_centers=untracked_face_centers,
            used_untracked_face_keys=used_untracked_face_keys,
        )

    # 현재 reenact 경로는 avatar bank를 전제로 하므로,
    # 얼굴별 avatar_id를 정하고 현재 yaw에 맞는 source_view를 항상 고른다.
    # source_view는 정면/좌측/우측 같은 avatar 기준 이미지 방향을 의미한다.
    avatar_id = choose_avatar_id_for_face(
        face_key=face_key,
        avatar_ids=avatar_ids,
        assignment_by_face=avatar_assignment_by_face,
        rng=avatar_rng,
    )
    avatar_profile = load_avatar_profile_for_id(avatar_id)
    source_view = select_avatar_view(avatar_profile, coeff_to_pose_radians(coeff_264)["yaw"])

    plan = FrameFacePlan(
        frame_index=frame_index,
        face_key=face_key,
        tracking_id=tracking_id_int,
        bbox_xyxy=tuple(float(v) for v in det_bbox_xyxy),
        coeff_264=np.asarray(coeff_264, dtype=np.float32),
        avatar_id=avatar_id,
        source_view=source_view,
    )
    return plan


def build_frame_plans(
    raw_frames: list[Any],
    *,
    excluded_tracking_ids: set[int],
    avatar_ids: list[str],
    load_avatar_profile_for_id: Callable[[str], dict[str, Any]],
    avatar_assignment_by_face: dict[str, str],
    avatar_rng: random.Random,
    should_process_frame_index: Callable[[int], bool],
) -> list[list[FrameFacePlan]]:
    # metadata를 한 번 읽으면서 "각 프레임에서 어떤 얼굴을 처리할지" 미리 정한다.
    # 이 단계에서는 실제 이미지 합성이나 warp 계산을 하지 않는다.
    # 프레임별 metadata에서 얼굴 후보를 고르고, bbox 보정/smoothing/avatar 선택까지만 수행한다.
    # 결과는 frame_plans[frame_index] 형태로 저장되며, 이후 keyframe 계산 단계에서 그대로 사용된다.
    smooth_state: dict[int, np.ndarray] = {}
    untracked_face_centers: dict[str, np.ndarray] = {}
    frame_plans: list[list[FrameFacePlan]] = []

    for frame_index, metadata_frame in enumerate(raw_frames):
        # frame_step 등으로 샘플링 대상이 아닌 프레임은 빈 plan만 넣고 넘어간다.
        # 이렇게 해야 frame_plans의 index가 원본 frame_index와 계속 일치한다.
        if not should_process_frame_index(frame_index):
            frame_plans.append([])
            continue
        if not isinstance(metadata_frame, Mapping):
            frame_plans.append([])
            continue

        # metadata에서 얼굴 목록을 꺼낸 뒤, 제외할 tracking_id가 있으면 먼저 제거한다.
        faces = filter_faces(
            metadata_frame.get("faces", []) or [],
            excluded_tracking_ids=excluded_tracking_ids,
        )
        # 현재 reenact 경로는 항상 모든 얼굴을 처리하므로 bbox 큰 순서대로 모두 선택한다.
        selected_faces = sorted(faces, key=bbox_area, reverse=True)

        plans_for_frame: list[FrameFacePlan] = []
        used_untracked_face_keys: set[str] = set()
        for slot_index, selected in enumerate(selected_faces):
            plan = build_single_face_plan(
                frame_index=frame_index,
                selected=selected,
                slot_index=slot_index,
                smooth_state=smooth_state,
                untracked_face_centers=untracked_face_centers,
                used_untracked_face_keys=used_untracked_face_keys,
                avatar_ids=avatar_ids,
                load_avatar_profile_for_id=load_avatar_profile_for_id,
                avatar_assignment_by_face=avatar_assignment_by_face,
                avatar_rng=avatar_rng,
            )
            if plan is None:
                continue
            plans_for_frame.append(plan)
        frame_plans.append(plans_for_frame)

    return frame_plans


def choose_keyframe_indices(frame_indices: list[int], warp_every: int) -> list[int]:
    # 특정 얼굴이 등장한 프레임 목록 중 실제 warp를 계산할 keyframe만 고른다.
    # warp_every가 1이면 모든 처리 대상 프레임을 keyframe으로 사용한다.
    # warp_every가 커지면 계산량은 줄지만, 중간 프레임은 이전 keyframe 결과를 더 많이 재사용하게 된다.
    # 첫 프레임과 마지막 프레임은 항상 포함해서 시작/끝 구간이 비는 일을 막는다.
    if not frame_indices:
        return []
    if warp_every <= 1:
        return list(frame_indices)

    chosen = {frame_indices[0], frame_indices[-1]}
    for available_index, frame_index in enumerate(frame_indices):
        if available_index % warp_every == 0:
            chosen.add(frame_index)
    return sorted(chosen)


def build_warped_keyframe(
    *,
    plan: FrameFacePlan,
    mean_face: np.ndarray,
    shape_basis: np.ndarray,
    blendshape_basis: np.ndarray,
    load_avatar_profile_for_id: Callable[[str], dict[str, Any]],
    avatar_view_cache: dict[str, dict[str, Any]],
    gpen_keyframe_restorer: Any,
    should_restore_keyframe: bool,
    key_restorer_mask_expand_px: int,
    key_restorer_feather_px: int,
) -> WarpedKeyframe:
    # FrameFacePlan 하나를 실제 warp 결과인 WarpedKeyframe 하나로 변환한다.
    # 이 함수가 keyframe 계산에서 가장 무거운 landmark 복원, avatar asset 로딩, warp_face를 담당한다.
    reconstructed = reconstruct_qualcomm_68_landmarks(
        plan.coeff_264,
        mean_face=mean_face,
        shape_basis=shape_basis,
        blendshape_basis=blendshape_basis,
    )
    target_landmarks_image = overlay_helpers.fit_landmarks_to_bbox(
        reconstructed.landmarks_2d,
        plan.bbox_xyxy,
        pad_left=DEFAULT_PAD_LEFT,
        pad_right=DEFAULT_PAD_RIGHT,
        pad_top=DEFAULT_PAD_TOP,
        pad_bottom=DEFAULT_PAD_BOTTOM,
    )
    target_points_crop = image_landmarks_to_crop_points(
        target_landmarks_image,
        plan.bbox_xyxy,
    )

    avatar_view_cache_key = f"{plan.avatar_id}:{plan.source_view}"
    source_assets = avatar_view_cache.get(avatar_view_cache_key)
    if source_assets is None:
        avatar_profile = load_avatar_profile_for_id(plan.avatar_id)
        source_assets = load_avatar_view_assets(avatar_profile, plan.source_view)
        avatar_view_cache[avatar_view_cache_key] = source_assets

    source_crop = np.asarray(source_assets["source_crop_bgr"], dtype=np.uint8)
    source_points = np.asarray(source_assets["source_points"], dtype=np.float32)

    reenacted_face, reenacted_mask = warp_face(source_crop, source_points, target_points_crop)
    if should_restore_keyframe:
        reenacted_face = restore_keyframe_face_region(
            reenacted_face,
            reenacted_mask,
            gpen_restorer=gpen_keyframe_restorer,
            mask_expand_px=key_restorer_mask_expand_px,
            feather_px=key_restorer_feather_px,
        )

    return WarpedKeyframe(
        frame_index=plan.frame_index,
        face_bgr=reenacted_face,
        mask_uint8=reenacted_mask,
        crop_points=np.asarray(target_points_crop, dtype=np.float32),
        avatar_id=plan.avatar_id,
        source_view=plan.source_view,
    )


def build_keyframe_cache(
    frame_plans: list[list[FrameFacePlan]],
    *,
    mean_face: np.ndarray,
    shape_basis: np.ndarray,
    blendshape_basis: np.ndarray,
    load_avatar_profile_for_id: Callable[[str], dict[str, Any]],
    gpen_keyframe_restorer: Any,
    warp_every: int,
    key_restorer_every: int,
    key_restorer_mask_expand_px: int,
    key_restorer_feather_px: int,
) -> tuple[dict[str, list[int]], dict[str, dict[int, WarpedKeyframe]]]:
    # FrameFacePlan 목록을 기반으로 실제 avatar 얼굴 warp 결과를 미리 계산한다.
    # 핵심 목적은 "모든 프레임에서 무거운 warp_face를 돌리지 않고", 선택된 keyframe에서만 계산하는 것이다.
    # 반환되는 cache_by_face는 face_key -> frame_index -> WarpedKeyframe 구조로 저장된다.
    # 렌더링 단계에서는 이 cache를 보고 현재 frame에 사용할 얼굴 이미지를 빠르게 가져온다.
    avatar_view_cache: dict[str, dict[str, Any]] = {}
    frames_by_face_key: dict[str, list[int]] = {}
    plan_lookup: dict[str, dict[int, FrameFacePlan]] = {}

    # 먼저 frame_plans를 얼굴 단위로 다시 묶는다.
    # 그래야 각 얼굴마다 등장한 프레임 목록을 만들고, 그 안에서 keyframe index를 고를 수 있다.
    for plans_for_frame in frame_plans:
        for plan in plans_for_frame:
            frames_by_face_key.setdefault(plan.face_key, []).append(plan.frame_index)
            plan_lookup.setdefault(plan.face_key, {})[plan.frame_index] = plan

    # 얼굴별로 keyframe을 고르고, 선택된 keyframe마다 실제 warp 결과를 만든다.
    keyframe_indices_by_face: dict[str, list[int]] = {}
    cache_by_face: dict[str, dict[int, WarpedKeyframe]] = {}

    for face_key, frame_indices in frames_by_face_key.items():
        chosen_indices = choose_keyframe_indices(frame_indices, warp_every)
        keyframe_indices_by_face[face_key] = chosen_indices
        cache_for_face: dict[int, WarpedKeyframe] = {}
        restorer_stride = max(1, int(key_restorer_every))
        for keyframe_order, frame_index in enumerate(chosen_indices):
            plan = plan_lookup[face_key][frame_index]
            # GPEN 같은 얼굴 복원기는 비용이 크므로 모든 keyframe에 적용하지 않고 stride 간격으로 적용한다.
            should_restore_keyframe = (keyframe_order % restorer_stride) == 0
            cache_for_face[frame_index] = build_warped_keyframe(
                plan=plan,
                mean_face=mean_face,
                shape_basis=shape_basis,
                blendshape_basis=blendshape_basis,
                load_avatar_profile_for_id=load_avatar_profile_for_id,
                avatar_view_cache=avatar_view_cache,
                gpen_keyframe_restorer=gpen_keyframe_restorer,
                should_restore_keyframe=should_restore_keyframe,
                key_restorer_mask_expand_px=key_restorer_mask_expand_px,
                key_restorer_feather_px=key_restorer_feather_px,
            )
        cache_by_face[face_key] = cache_for_face

    return keyframe_indices_by_face, cache_by_face


def resolve_causal_warp(
    frame_index: int,
    keyframe_indices: list[int],
    keyframe_cache: Mapping[int, WarpedKeyframe],
    *,
    transition_frames: int,
) -> WarpedKeyframe:
    # 현재 frame_index에서 사용할 WarpedKeyframe을 고른다.
    # causal 방식이므로 미래 keyframe은 보지 않고, 현재 frame 이전에 계산된 keyframe만 사용한다.
    # transition_frames 범위 안에서는 이전 keyframe과 현재 keyframe을 부드럽게 섞어 갑작스러운 변화감을 줄인다.
    if not keyframe_indices:
        raise RuntimeError("No keyframes available for interpolation.")

    # bisect_right로 현재 frame_index 이하에서 가장 가까운 keyframe 위치를 찾는다.
    position = bisect.bisect_right(keyframe_indices, frame_index) - 1
    if position < 0:
        return keyframe_cache[keyframe_indices[0]]

    current_index = keyframe_indices[position]
    current_keyframe = keyframe_cache[current_index]
    # 현재 프레임이 keyframe 그 자체라면 보간하지 않고 현재 keyframe을 그대로 쓴다.
    if frame_index == current_index:
        return current_keyframe

    # 보간을 쓰지 않거나 첫 keyframe이면 비교할 이전 keyframe이 없으므로 현재 keyframe을 그대로 사용한다.
    if transition_frames <= 0 or position == 0:
        return current_keyframe

    previous_index = keyframe_indices[position - 1]
    gap = current_index - previous_index
    if gap <= 1:
        return current_keyframe

    age_since_keyframe = frame_index - current_index
    if age_since_keyframe < 0 or age_since_keyframe >= transition_frames:
        return current_keyframe

    # alpha가 0에 가까울수록 이전 keyframe에 가깝고, 1에 가까울수록 현재 keyframe에 가깝다.
    # face, mask, crop_points를 같은 비율로 섞어 이미지와 위치가 함께 부드럽게 넘어가도록 한다.
    previous_keyframe = keyframe_cache[previous_index]
    alpha = (age_since_keyframe + 1) / (transition_frames + 1)
    face = (
        previous_keyframe.face_bgr.astype(np.float32) * (1.0 - alpha)
        + current_keyframe.face_bgr.astype(np.float32) * alpha
    )
    mask = (
        previous_keyframe.mask_uint8.astype(np.float32) * (1.0 - alpha)
        + current_keyframe.mask_uint8.astype(np.float32) * alpha
    )
    crop_points = (
        previous_keyframe.crop_points.astype(np.float32) * (1.0 - alpha)
        + current_keyframe.crop_points.astype(np.float32) * alpha
    )
    return WarpedKeyframe(
        frame_index=frame_index,
        face_bgr=np.clip(face, 0, 255).astype(np.uint8),
        mask_uint8=np.clip(mask, 0, 255).astype(np.uint8),
        crop_points=np.asarray(crop_points, dtype=np.float32),
        avatar_id=current_keyframe.avatar_id,
        source_view=previous_keyframe.source_view if alpha < 0.5 else current_keyframe.source_view,
    )
