from __future__ import annotations

# source portrait를 미리 분석하고 저장할 때 쓰는 준비 단계 유틸 모음이다.
# MediaPipe bbox 검출, FaceMap coeff 추론, landmark 복원처럼
# 상대적으로 무거운 전처리/모델 의존성은 runtime 모듈과 분리한다.

import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.converters.coeffs_to_arkit52 import reconstruct_qualcomm_68_landmarks

MODULE_DIR = Path(__file__).resolve().parent
CROP_SIZE = 256
MEDIAPIPE_TASK_FILENAME = "face_landmarker.task"
MODEL_INPUT_SIZE = 128


def _resolve_mediapipe_task_path(task_path: str | None) -> Path:
    if task_path:
        resolved = Path(task_path).expanduser().resolve()
        if not resolved.exists():
            raise RuntimeError(f"MediaPipe task file was not found: {resolved}")
        return resolved

    candidate_dirs = [
        MODULE_DIR / "models",
        ROOT_DIR / "models",
    ]
    for candidate_dir in candidate_dirs:
        candidate = candidate_dir / MEDIAPIPE_TASK_FILENAME
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(str(candidate_dir / MEDIAPIPE_TASK_FILENAME) for candidate_dir in candidate_dirs)
    raise RuntimeError(
        "MediaPipe task file was not found. Place it at one of:\n"
        f"{searched}\n"
        "or pass an explicit local path."
    )


def ensure_facemap_model() -> Any:
    try:
        from qai_hub_models.models.facemap_3dmm.model import FaceMap_3DMM
    except ImportError as exc:
        raise RuntimeError(
            "qai_hub_models is required. Install it first, for example:\n"
            "pip install qai-hub-models torch opencv-python scipy mediapipe"
        ) from exc
    model = FaceMap_3DMM.from_pretrained()
    model.eval()
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def load_qualcomm_asset_array(path: str | None, asset_name: str) -> np.ndarray:
    if path:
        candidate = Path(path).expanduser()
        if candidate.exists():
            return np.load(candidate)

    tools_dir = Path(__file__).resolve().parent
    repo_root = tools_dir.parent
    bundled_candidates = [
        tools_dir / asset_name,
        tools_dir / "assets" / asset_name,
        repo_root / asset_name,
        repo_root / "assets" / asset_name,
        repo_root / "shared" / "facemap_assets" / asset_name,
        repo_root / "facemap_assets" / asset_name,
        repo_root / "qualcomm_assets" / asset_name,
    ]
    for candidate in bundled_candidates:
        if candidate.exists():
            return np.load(candidate)

    try:
        from qai_hub_models.models.facemap_3dmm.utils import CachedWebModelAsset, load_numpy
    except ImportError as exc:
        missing = path if path else f"<auto:{asset_name}>"
        raise RuntimeError(
            f"Failed to load {asset_name} from {missing}. "
            "Install qai-hub-models or pass a valid local .npy path."
        ) from exc

    model_id = "facemap_3dmm"
    model_asset_version = 1
    return load_numpy(
        CachedWebModelAsset.from_asset_store(model_id, model_asset_version, asset_name)
    )


def ensure_face_landmarker(task_path: str | None) -> Any:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required for automatic source face detection. "
            "Install it with: pip install mediapipe"
        ) from exc

    resolved_task_path = _resolve_mediapipe_task_path(task_path)
    base_options = python.BaseOptions(model_asset_path=str(resolved_task_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def detect_source_bbox_with_mediapipe(
    image_bgr: np.ndarray,
    *,
    landmarker: Any,
    mp_module: Any,
    pad_ratio: float,
) -> list[int]:
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        raise RuntimeError("No face found in source image.")

    pts = np.asarray([[pt.x * w, pt.y * h] for pt in result.face_landmarks[0]], dtype=np.float32)
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    bw = x2 - x1
    bh = y2 - y1
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    size = max(bw, bh) * (1.0 + pad_ratio)
    half = 0.5 * size
    return [
        max(0, int(round(cx - half))),
        max(0, int(round(cy - half))),
        min(w, int(round(cx + half))),
        min(h, int(round(cy + half))),
    ]


def crop_square(image_bgr: np.ndarray, bbox_xyxy: Sequence[float], crop_size: int = CROP_SIZE) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError(f"Invalid crop bbox: {bbox_xyxy}")
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)


def to_model_input(crop_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(crop_bgr, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device)


def infer_coeff(model: Any, crop_bgr: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model_input = to_model_input(crop_bgr, device)
    with torch.no_grad():
        coeff = model(model_input).detach().cpu().numpy()[0]
    if coeff.shape[0] > 264:
        coeff = coeff[:264]
    if coeff.shape[0] != 264:
        raise RuntimeError(f"Expected FaceMap output length 264, got {coeff.shape[0]}")
    return coeff.astype(np.float32)


def coeff_to_crop_landmarks(
    coeff: Sequence[float] | np.ndarray,
    *,
    mean_face: np.ndarray,
    shape_basis: np.ndarray,
    blendshape_basis: np.ndarray,
    crop_size: int = CROP_SIZE,
) -> np.ndarray:
    reconstructed = reconstruct_qualcomm_68_landmarks(
        coeff,
        mean_face=mean_face,
        shape_basis=shape_basis,
        blendshape_basis=blendshape_basis,
    )
    points = reconstructed.landmarks_2d.astype(np.float32)
    points[:, 0] = (points[:, 0] + MODEL_INPUT_SIZE * 0.5) * crop_size / MODEL_INPUT_SIZE
    points[:, 1] = (points[:, 1] + MODEL_INPUT_SIZE * 0.5) * crop_size / MODEL_INPUT_SIZE
    return points


__all__ = [
    "CROP_SIZE",
    "coeff_to_crop_landmarks",
    "crop_square",
    "detect_source_bbox_with_mediapipe",
    "ensure_face_landmarker",
    "ensure_facemap_model",
    "infer_coeff",
    "load_qualcomm_asset_array",
]
