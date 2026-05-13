from __future__ import annotations

# 1. 입력 metadata / video / avatar bank 경로를 읽는다.
# 2. avatar bank 안에서 어떤 avatar_id 후보가 있는지 찾는다.
# 3. metadata를 한 번 훑으면서 프레임별 처리 계획(frame_plans)을 만든다.
# 4. 모든 프레임을 직접 warp하지 않고, keyframe만 먼저 계산해서 cache로 저장한다.
# 5. 실제 video를 다시 읽으면서 각 프레임에서 어떤 keyframe 결과를 쓸지 고른다.
# 6. warp된 얼굴을 원본 frame 위에 합성하고 output video로 저장한다.

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

import metadata_bbox_utils as overlay_helpers
from reenact_composite import composite_face
from reenact_restore import load_gpen_keyframe_restorer
from reenact_keyframe_cache import (
    build_frame_plans,
    build_keyframe_cache,
    resolve_causal_warp,
)
from reenact_assets_runtime import (
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


def should_process_frame_index(frame_index: int, frame_step: int) -> bool:
    # 처리해야하는 프레임 계산
    return frame_step <= 1 or frame_index % frame_step == 0

def resolve_output_fps(input_fps: float, frame_step: int) -> float:
    # 출력 영상 fps 정하기
    base_fps = float(input_fps) if input_fps and input_fps > 0 else 30.0
    return max(base_fps / max(1, int(frame_step)), 1e-6)


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


def run_keyframe_reenact_pipeline(args: argparse.Namespace) -> None:
    # 프레임 설정 값
    warp_every = 1
    transition_frames = 1
    frame_step = 3

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

    # 아래 카운터들은 실행 결과를 간단히 요약하는 데 쓴다.
    frame_index = 0
    frames_sampled = 0
    frames_composited = 0

    try:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {args.video}")

        # 출력 경로가 아직 없으면 부모 폴더까지 만들어 둔다.
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

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

            # frame_plans는 해당 프레임에서 처리할 얼굴을 인지
            plans_for_frame = frame_plans[frame_index] if frame_index < len(frame_plans) else []
            for plan in plans_for_frame:
                # 같은 face_key에 대해 어떤 frame들이 keyframe인지와, 그 keyframe 계산 결과가 무엇인지 가져온다.
                keyframe_indices = keyframe_indices_by_face.get(plan.face_key, [])
                keyframe_cache = keyframe_cache_by_face.get(plan.face_key, {})
                if not keyframe_indices or not keyframe_cache:
                    continue

                # 현재 frame에서 사용할 가장 적절한 keyframe 결과를 고른다.
                # 필요하면 이전/현재 keyframe 결과를 transition_frames 만큼 섞는다.
                warped = resolve_causal_warp(
                    frame_index,
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
                if args.draw_bbox:
                    overlay_helpers.draw_bbox_overlay(
                        frame_bgr,
                        plan.bbox_xyxy,
                        tracking_id=plan.tracking_id,
                        line_thickness=args.line_thickness,
                        hide_labels=args.hide_labels,
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

    # 실행 확인
    print(
        json.dumps(
            {
                "frames_sampled": frames_sampled,
                "frames_composited": frames_composited,
                "keyframes_computed": int(sum(len(v) for v in keyframe_indices_by_face.values())),
                "gpen_load_error": gpen_load_error,
                "output_video": str(output_video_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
