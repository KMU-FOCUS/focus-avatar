from __future__ import annotations

# 1. 입력 metadata / video / avatar bank 경로를 읽는다.
# 2. avatar bank 안에서 어떤 avatar_id 후보가 있는지 찾는다.
# 3. metadata를 한 번 훑으면서 프레임별 처리 계획(frame_plans)을 만든다.
# 4. 모든 프레임을 직접 warp하지 않고, keyframe만 먼저 계산해서 cache로 저장한다.
# 5. 실제 video를 다시 읽으면서 각 프레임에서 어떤 keyframe 결과를 쓸지 고른다.
# 6. warp된 얼굴을 원본 frame 위에 합성하고 output video로 저장한다.

import argparse
import bisect
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from . import metadata_bbox_utils as overlay_helpers
from .reenact_composite import build_debug_face_mask, composite_face
from .reenact_restore import load_gpen_keyframe_restorer
from .reenact_keyframe_cache import (
    build_frame_plans,
    build_keyframe_cache,
    resolve_causal_warp,
)
from .reenact_assets_runtime import (
    discover_avatar_bank_entries,
    load_avatar_profile_by_id,
    load_json,
)

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent

# facemap basis 파일은 여기에서 고정 경로로 관리한다.
# 위치를 바꾸고 싶으면 이 매핑만 수정하면 된다.
FACEMAP_ASSET_PATHS = {
    "meanFace.npy": REPO_ROOT / "shared" / "facemap_assets" / "meanFace.npy",
    "shapeBasis.npy": REPO_ROOT / "shared" / "facemap_assets" / "shapeBasis.npy",
    "blendShape.npy": REPO_ROOT / "shared" / "facemap_assets" / "blendShape.npy",
}


def crop_points_to_image_landmarks(
    crop_points: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    *,
    crop_size: int = 256,
) -> np.ndarray:
    # 디버그 코드:
    # - warp용 crop_points를 다시 이미지 좌표로 되돌려,
    #   출력 영상 위에 landmark 점을 확인할 때만 쓴다.
    x1, y1, x2, y2 = map(float, bbox_xyxy[:4])
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    points = np.asarray(crop_points, dtype=np.float32).reshape(-1, 2).copy()
    points[:, 0] = points[:, 0] / float(crop_size) * width + x1
    points[:, 1] = points[:, 1] / float(crop_size) * height + y1
    return points


def overlay_debug_mask(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    mask_uint8: np.ndarray,
    *,
    alpha: float,
) -> None:
    # 디버그 코드:
    # - 최종 합성에 실제로 쓰인 mask를 반투명 overlay로 보여주는 전용 함수다.
    # - 결과 분석용이므로, 제거해도 합성 파이프라인의 품질 로직에는 영향이 없다.
    x1, y1, x2, y2 = bbox_xyxy
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return
    alpha_value = max(0.0, min(1.0, float(alpha)))
    overlay = roi.copy()
    overlay[..., 2] = np.maximum(overlay[..., 2], mask_uint8)
    blended = cv2.addWeighted(overlay, alpha_value, roi, 1.0 - alpha_value, 0.0)
    roi[:] = blended

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(roi, contours, -1, (0, 255, 255), thickness=1, lineType=cv2.LINE_AA)


def should_process_frame_index(frame_index: int, frame_step: int) -> bool:
    # 처리해야하는 프레임 계산
    return frame_step <= 1 or frame_index % frame_step == 0

def resolve_output_fps(input_fps: float, frame_step: int) -> float:
    # 출력 영상 fps 정하기
    base_fps = float(input_fps) if input_fps and input_fps > 0 else 30.0
    return max(base_fps / max(1, int(frame_step)), 1e-6)


def build_sampled_metadata_pts_index(
    raw_frames: list[Any],
    *,
    frame_step: int,
) -> tuple[list[int], list[int]] | None:
    # sampled metadata frame들에 대해 relative pts_us 인덱스를 만든다.
    # pts_us가 빠진 frame이 하나라도 있으면 기존 frame_index 정렬로 fallback 한다.
    sampled_indices = [frame_index for frame_index in range(len(raw_frames)) if should_process_frame_index(frame_index, frame_step)]
    if not sampled_indices:
        return None

    sampled_pts_us: list[int] = []
    for frame_index in sampled_indices:
        frame = raw_frames[frame_index]
        if not isinstance(frame, Mapping):
            return None
        pts_us = frame.get("pts_us")
        if pts_us is None:
            return None
        try:
            sampled_pts_us.append(int(pts_us))
        except (TypeError, ValueError):
            return None

    start_pts_us = sampled_pts_us[0]
    sampled_relative_pts_us = [max(0, pts_us - start_pts_us) for pts_us in sampled_pts_us]
    return sampled_indices, sampled_relative_pts_us


def probe_video_relative_pts_us(video_path: str | Path) -> list[int] | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time,pts_time,pkt_dts_time,pkt_pts_time",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    payload = json.loads(result.stdout)
    frame_items = payload.get("frames")
    if not isinstance(frame_items, list) or not frame_items:
        return None

    raw_pts_us: list[int] = []
    for item in frame_items:
        if not isinstance(item, Mapping):
            return None
        pts_us: int | None = None
        for key in ("best_effort_timestamp_time", "pts_time", "pkt_dts_time", "pkt_pts_time"):
            value = item.get(key)
            if value in (None, "N/A"):
                continue
            try:
                pts_us = int(round(float(value) * 1_000_000.0))
            except (TypeError, ValueError):
                pts_us = None
            if pts_us is not None:
                break
        if pts_us is None:
            return None
        raw_pts_us.append(pts_us)

    start_pts_us = raw_pts_us[0]
    return [max(0, pts_us - start_pts_us) for pts_us in raw_pts_us]


def probe_has_audio_stream(video_path: str | Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return False

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return False
    stream_items = payload.get("streams")
    return isinstance(stream_items, list) and len(stream_items) > 0


def resolve_metadata_frame_index_for_video_time(
    current_video_us: int,
    sampled_metadata_indices: list[int],
    sampled_relative_pts_us: list[int],
) -> int:
    # 현재 비디오 시점과 가장 가까운 sampled metadata frame_index를 찾는다.
    position = bisect.bisect_left(sampled_relative_pts_us, current_video_us)
    if position <= 0:
        return sampled_metadata_indices[0]
    if position >= len(sampled_metadata_indices):
        return sampled_metadata_indices[-1]

    previous_pts_us = sampled_relative_pts_us[position - 1]
    current_pts_us = sampled_relative_pts_us[position]
    if abs(current_video_us - previous_pts_us) <= abs(current_pts_us - current_video_us):
        return sampled_metadata_indices[position - 1]
    return sampled_metadata_indices[position]


def load_required_facemap_assets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing_paths = [
        f"{asset_name}: {asset_path}"
        for asset_name, asset_path in FACEMAP_ASSET_PATHS.items()
        if not asset_path.exists()
    ]
    if missing_paths:
        missing_text = "\n".join(missing_paths)
        raise RuntimeError(
            "Missing required facemap assets. Update FACEMAP_ASSET_PATHS in "
            f"{Path(__file__).name} or place the files at:\n{missing_text}"
        )

    mean_face = np.load(FACEMAP_ASSET_PATHS["meanFace.npy"], allow_pickle=False)
    shape_basis = np.load(FACEMAP_ASSET_PATHS["shapeBasis.npy"], allow_pickle=False)
    blendshape_basis = np.load(FACEMAP_ASSET_PATHS["blendShape.npy"], allow_pickle=False)
    return mean_face, shape_basis, blendshape_basis


def remux_original_audio(
    *,
    video_only_path: str | Path,
    source_video_path: str | Path,
    output_video_path: str | Path,
) -> tuple[bool, str | None]:
    if not probe_has_audio_stream(source_video_path):
        return False, "source video has no audio stream"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_only_path),
        "-i",
        str(source_video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return False, stderr or "ffmpeg remux failed"
    return True, None


def run_keyframe_reenact_pipeline(args: argparse.Namespace) -> None:
    # 프레임 설정 값
    warp_every = 1
    transition_frames = 1
    frame_step = 1

    if warp_every < 1:
        raise RuntimeError("warp_every must be at least 1.")
    if frame_step < 1:
        raise RuntimeError("frame_step must be at least 1.")

    # avatar_bank_dir는 "어떤 avatar 후보들을 쓸 수 있는지" 알려주는 입력이다.
    # 이후에는 여기서 발견한 avatar_id들 중 하나가 얼굴마다 배정된다.
    avatar_bank_inputs = [str(path) for path in args.avatar_bank_dir]
    if not avatar_bank_inputs:
        raise RuntimeError("Pass --avatar-bank-dir with one or more avatar bank directories.")

    # metadata는 프레임별 얼굴 bbox / tracking_id / coeff를 담고 있는 driving 정보다.
    metadata = load_json(args.metadata)
    if not isinstance(metadata, Mapping):
        raise ValueError("Expected metadata JSON object with a top-level 'frames' array.")
    raw_frames = metadata.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError("Expected metadata JSON with a top-level 'frames' array.")

    # 지금 구조에서는 모든 얼굴을 처리하고, 특정 tracking id를 따로 제외하지 않는다.
    # 형식은 남겨두되 실제 값은 빈 set이다.
    excluded_tracking_ids: set[int] = set()

    # Qualcomm coeff -> landmark 복원에 필요한 basis 배열들이다.
    # 경로는 FACEMAP_ASSET_PATHS에서 고정 관리하고, 시작 시점에 즉시 검증한다.
    mean_face, shape_basis, blendshape_basis = load_required_facemap_assets()

    # face_key -> avatar_id 매핑을 저장한다.
    # 같은 얼굴이 다시 나오면 처음 배정된 avatar를 계속 유지해야 하므로 이 dict가 필요하다.
    avatar_assignment_by_face: dict[str, str] = {}

    # random seed를 고정하면 "얼굴 A가 어떤 avatar를 받는지"가 실행마다 재현된다.
    avatar_rng = random.Random(int(args.avatar_random_seed))

    # GPEN은 선택 기능이라, 로드 실패가 전체 reenact를 깨지 않도록 분리해서 처리한다.
    gpen_keyframe_restorer = None
    gpen_load_error: str | None = None

    # avatar bank 루트 폴더를 스캔해서,
    # "어떤 avatar_id를 쓸 수 있는지"와 그 profile.json 경로를 만든다.
    avatar_profile_paths_by_id = discover_avatar_bank_entries(avatar_bank_inputs)
    avatar_ids = sorted(avatar_profile_paths_by_id.keys())
    if not avatar_ids:
        raise RuntimeError("No avatar folders with profile.json were found under --avatar-bank-dir inputs.")

    # profile.json은 처음부터 전부 읽지 않고, 실제로 필요한 avatar_id만 지연 로드한다.
    # 이 dict는 그 지연 로드 결과를 메모리에 재사용하는 캐시다.
    avatar_profile_cache: dict[str, dict[str, Any]] = {}

    def load_avatar_profile_for_id(avatar_id: str) -> dict[str, Any]:
        # 같은 avatar_id를 여러 프레임에서 반복해서 쓰므로,
        # profile.json을 매번 다시 읽지 않도록 한 번 읽은 결과를 재사용한다.
        cached_profile = avatar_profile_cache.get(avatar_id)
        if cached_profile is not None:
            return cached_profile
        profile = load_avatar_profile_by_id(avatar_profile_paths_by_id, avatar_id)
        avatar_profile_cache[avatar_id] = profile
        return profile

    if args.gpen_model:
        try:
            gpen_keyframe_restorer = load_gpen_keyframe_restorer(
                str(Path(args.gpen_model).expanduser().resolve()),
                provider=str(args.gpen_provider),
                input_size=int(args.gpen_input_size),
            )
        except Exception as exc:
            gpen_load_error = str(exc)
            print(f"[warn] GPEN disabled: {exc}")

    # 1:
    # metadata만 읽어서 "각 frame에서 어떤 얼굴을 어떤 bbox로 처리할지" 계획한다.
    # 현재 reenact 경로는 항상 모든 얼굴을 처리하므로 dominant track 선택은 쓰지 않는다.
    frame_plans = build_frame_plans(
        raw_frames,
        excluded_tracking_ids=excluded_tracking_ids,
        avatar_ids=avatar_ids,
        load_avatar_profile_for_id=load_avatar_profile_for_id,
        avatar_assignment_by_face=avatar_assignment_by_face,
        avatar_rng=avatar_rng,
        should_process_frame_index=lambda frame_index: should_process_frame_index(frame_index, frame_step),
    )
    metadata_pts_index = build_sampled_metadata_pts_index(
        raw_frames,
        frame_step=frame_step,
    )

    # 2:
    # keyframe만 먼저 계산해서 cache에 저장한다.
    # 이후 실제 video loop에서는 이 cache를 꺼내 사용한다.
    keyframe_indices_by_face, keyframe_cache_by_face = build_keyframe_cache(
        frame_plans,
        mean_face=mean_face,
        shape_basis=shape_basis,
        blendshape_basis=blendshape_basis,
        load_avatar_profile_for_id=load_avatar_profile_for_id,
        gpen_keyframe_restorer=gpen_keyframe_restorer,
        warp_every=warp_every,
        key_restorer_every=args.key_restorer_every,
        key_restorer_mask_expand_px=args.key_restorer_mask_expand_px,
        key_restorer_feather_px=args.key_restorer_feather_px,
    )

    # 원본 비디오 읽기
    # 위에서 만든 frame_plans / keyframe_cache를 바탕으로 합성
    cap: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    output_video_path = Path(args.output_video).expanduser().resolve()
    final_output_video_path = output_video_path
    final_output_video_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_file = tempfile.NamedTemporaryFile(
        prefix=f"{output_video_path.stem}_video_only_",
        suffix=output_video_path.suffix or ".mp4",
        dir=str(output_video_path.parent),
        delete=False,
    )
    temp_output_file.close()
    output_video_path = Path(temp_output_file.name)
    video_relative_pts_us = probe_video_relative_pts_us(args.video)

    # 아래 카운터들은 실행 결과를 간단히 요약하는 데 쓴다.
    frame_index = 0
    frames_sampled = 0
    frames_composited = 0
    input_fps = 30.0
    width = 0
    height = 0
    audio_remuxed = False
    audio_remux_error: str | None = None

    try:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {args.video}")

        # sampled frame만 저장하므로 output fps도 줄여서 계산한다.
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        output_fps = resolve_output_fps(float(input_fps), frame_step)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create output video: {output_video_path}")

        # 3:
        # 실제 비디오 프레임을 읽으면서 현재 프레임에서 써야 하는 cached keyframe 결과를 골라 합성한다.
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # sampling 대상이 아닌 frame은 그냥 건너뛴다.
            if not should_process_frame_index(frame_index, frame_step):
                frame_index += 1
                continue

            # 현재 비디오 시간에 가장 가까운 metadata frame을 찾는다.
            # ffprobe frame timestamp를 우선 사용하고, 없으면 기존 OpenCV POS_MSEC로 fallback 한다.
            metadata_frame_index = frame_index
            if metadata_pts_index is not None:
                sampled_metadata_indices, sampled_relative_pts_us = metadata_pts_index
                if video_relative_pts_us is not None and frame_index < len(video_relative_pts_us):
                    current_video_us = int(video_relative_pts_us[frame_index])
                else:
                    current_video_us = int(round(max(0.0, cap.get(cv2.CAP_PROP_POS_MSEC)) * 1000.0))
                metadata_frame_index = resolve_metadata_frame_index_for_video_time(
                    current_video_us,
                    sampled_metadata_indices,
                    sampled_relative_pts_us,
                )

            # frame_plans는 해당 metadata 프레임에서 처리할 얼굴을 인지
            plans_for_frame = frame_plans[metadata_frame_index] if metadata_frame_index < len(frame_plans) else []
            for plan in plans_for_frame:
                # 같은 face_key에 대해 어떤 frame들이 keyframe인지와, 그 keyframe 계산 결과가 무엇인지 가져온다.
                keyframe_indices = keyframe_indices_by_face.get(plan.face_key, [])
                keyframe_cache = keyframe_cache_by_face.get(plan.face_key, {})
                if not keyframe_indices or not keyframe_cache:
                    continue

                # 현재 frame에서 사용할 가장 적절한 keyframe 결과를 고른다.
                # 필요하면 이전/현재 keyframe 결과를 transition_frames 만큼 섞는다.
                warped = resolve_causal_warp(
                    metadata_frame_index,
                    keyframe_indices,
                    keyframe_cache,
                    transition_frames=transition_frames,
                )

                # 실제 합성 단계:
                # 캐시된 reenacted face를 현재 frame의 bbox 위치에 붙인다.
                frame_bgr = composite_face(
                    frame_bgr,
                    warped.face_bgr,
                    plan.bbox_xyxy,
                    warped.crop_points,
                    face_mask_override=warped.mask_uint8,
                )

                # draw_bbox는 디버깅용 시각화라, 필요할 때만 overlay를 그린다.
                # 디버그 코드:
                # - 아래 draw_* 블록들은 출력 프레임 위에 진단용 overlay만 더한다.
                # - 실제 warp/composite 결과 자체는 이 옵션들과 무관하게 이미 계산돼 있다.
                if args.draw_bbox:
                    overlay_helpers.draw_bbox_overlay(
                        frame_bgr,
                        plan.bbox_xyxy,
                        tracking_id=plan.tracking_id,
                        line_thickness=args.line_thickness,
                        hide_labels=args.hide_labels,
                    )
                if args.draw_landmarks:
                    landmarks_image_xy = crop_points_to_image_landmarks(
                        warped.crop_points,
                        plan.bbox_xyxy,
                    )
                    overlay_helpers.draw_landmarks(
                        frame_bgr,
                        landmarks_image_xy,
                        radius=int(args.landmark_radius),
                        color=(40, 170, 255),
                    )
                if args.draw_mask:
                    debug_bbox, debug_mask = build_debug_face_mask(
                        frame_bgr.shape,
                        plan.bbox_xyxy,
                        warped.crop_points,
                        face_mask_override=warped.mask_uint8,
                    )
                    if debug_bbox is not None and debug_mask is not None:
                        overlay_debug_mask(
                            frame_bgr,
                            debug_bbox,
                            debug_mask,
                            alpha=float(args.mask_alpha),
                        )
                frames_composited += 1

            # 현재 출력 프레임 1장을 비디오 파일에 기록
            writer.write(frame_bgr)
            frames_sampled += 1
            frame_index += 1
    finally:
        # 파일 핸들 정리
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()

    try:
        audio_remuxed, audio_remux_error = remux_original_audio(
            video_only_path=output_video_path,
            source_video_path=args.video,
            output_video_path=final_output_video_path,
        )
        if not audio_remuxed:
            output_video_path.replace(final_output_video_path)
    finally:
        if output_video_path.exists() and output_video_path != final_output_video_path:
            output_video_path.unlink(missing_ok=True)

    # 실행 확인
    print(
        json.dumps(
            {
                "frames_sampled": frames_sampled,
                "frames_composited": frames_composited,
                "keyframes_computed": int(sum(len(v) for v in keyframe_indices_by_face.values())),
                "gpen_load_error": gpen_load_error,
                "audio_remuxed": audio_remuxed,
                "audio_remux_error": audio_remux_error,
                "pts_source": "ffprobe" if video_relative_pts_us is not None else "opencv_pos_msec",
                "output_video": str(final_output_video_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
