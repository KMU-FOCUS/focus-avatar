#!/usr/bin/env python3
"""Precompute reusable source assets for Qualcomm metadata reenactment."""

from __future__ import annotations

# reenact에서 재사용할 source portrait 자산을 미리 계산하는 스크립트다.
# 안정적인 얼굴 crop, 추론된 coefficient, source landmark를 한 번만 뽑아 두고,
# 이후 실행에서 같은 portrait 분석을 반복하지 않게 한다.

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .source_avatar_prepare import (
        CROP_SIZE,
        coeff_to_crop_landmarks,
        crop_square,
        detect_source_bbox_with_mediapipe,
        ensure_face_landmarker,
        ensure_facemap_model,
        infer_coeff,
        load_qualcomm_asset_array,
    )
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise RuntimeError(
            "torch is required for Qualcomm FaceMap 3DMM inference. "
            "Install it in the current Python environment first."
        ) from exc
    raise

DEFAULT_MEAN_FACE_PATH: str | None = None
DEFAULT_SHAPE_BASIS_PATH: str | None = None
DEFAULT_BLENDSHAPE_BASIS_PATH: str | None = None
DEFAULT_SOURCE_PAD_RATIO = 0.35
DEFAULT_LANDMARKER_TASK: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve_source_bbox(
    source_image: np.ndarray,
    *,
    source_pad_ratio: float,
    landmarker_task: str | None,
) -> np.ndarray:
    # builder 체인에서는 source bbox를 따로 주지 않고 자동 검출만 사용한다.
    mp_module, landmarker = ensure_face_landmarker(landmarker_task)
    return np.asarray(
        detect_source_bbox_with_mediapipe(
            source_image,
            landmarker=landmarker,
            mp_module=mp_module,
            pad_ratio=float(source_pad_ratio),
        ),
        dtype=np.float32,
    )


def build_source_asset(
    *,
    source_image_path: str | Path,
    mean_face_path: str | None = None,
    shape_basis_path: str | None = None,
    blendshape_basis_path: str | None = None,
    source_pad_ratio: float = 0.35,
    landmarker_task: str | None = None,
) -> dict[str, Any]:
    # source portrait 하나에 대한 핵심 precompute 단계다.
    # 1) portrait 로드
    # 2) 정사각형에 가까운 얼굴 bbox 결정
    # 3) crop 후 FaceMap 추론
    # 4) crop 좌표계에서 source landmark 복원
    source_path = Path(source_image_path).expanduser().resolve()
    source_image = cv2.imread(str(source_path))
    if source_image is None:
        raise RuntimeError(f"Failed to read source image: {source_path}")

    source_bbox_xyxy = resolve_source_bbox(
        source_image,
        source_pad_ratio=source_pad_ratio,
        landmarker_task=landmarker_task,
    )

    crop_size = int(CROP_SIZE)
    source_crop = crop_square(source_image, source_bbox_xyxy, crop_size=crop_size)
    model = ensure_facemap_model()
    mean_face = load_qualcomm_asset_array(mean_face_path, "meanFace.npy")
    shape_basis = load_qualcomm_asset_array(shape_basis_path, "shapeBasis.npy")
    blendshape_basis = load_qualcomm_asset_array(blendshape_basis_path, "blendShape.npy")

    source_coeff = infer_coeff(model, source_crop)
    source_points = coeff_to_crop_landmarks(
        source_coeff,
        mean_face=mean_face,
        shape_basis=shape_basis,
        blendshape_basis=blendshape_basis,
    )

    return {
        "source_image_path": str(source_path),
        "source_bbox_xyxy": source_bbox_xyxy.astype(np.float32),
        "source_crop_bgr": source_crop.astype(np.uint8),
        "source_coeff_264": source_coeff.astype(np.float32),
        "source_points": source_points.astype(np.float32),
        "crop_size": crop_size,
    }


def save_source_asset(
    *,
    asset: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    # 이후 reenact 실행이 실제로 필요로 하는 값만 저장한다.
    # 이 함수의 실제 산출물은 .npz 하나이며, 반환 summary는 호출부 로그용이다.
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        source_bbox_xyxy=asset["source_bbox_xyxy"].astype(np.float32),
        source_crop_bgr=asset["source_crop_bgr"].astype(np.uint8),
        source_coeff_264=asset["source_coeff_264"].astype(np.float32),
        source_points=asset["source_points"].astype(np.float32),
        crop_size=np.asarray([asset["crop_size"]], dtype=np.int32),
        source_image_path=np.asarray([asset["source_image_path"]]),
    )

    summary = {
        "output": str(output_path),
        "source_image": asset["source_image_path"],
        "source_bbox_xyxy": asset["source_bbox_xyxy"].astype(float).tolist(),
        "crop_size": int(asset["crop_size"]),
        "source_points_shape": list(asset["source_points"].shape),
        "source_coeff_shape": list(asset["source_coeff_264"].shape),
    }

    return summary


def main() -> None:
    args = parse_args()
    # CLI 진입점은 의도적으로 작게 유지한다.
    # 한 번 계산하고, 한 번 저장하고, 요약만 출력한다.
    asset = build_source_asset(
        source_image_path=args.source_image,
        mean_face_path=DEFAULT_MEAN_FACE_PATH,
        shape_basis_path=DEFAULT_SHAPE_BASIS_PATH,
        blendshape_basis_path=DEFAULT_BLENDSHAPE_BASIS_PATH,
        source_pad_ratio=float(DEFAULT_SOURCE_PAD_RATIO),
        landmarker_task=DEFAULT_LANDMARKER_TASK,
    )
    summary = save_source_asset(
        asset=asset,
        output_path=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()