from __future__ import annotations

# reenact keyframe 복원용 헬퍼 모음이다.
# GPEN 로딩 디테일을 숨기고,
# warp된 얼굴 crop을 재사용하기 전에 복원하는 공통 인터페이스를 제공한다.

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

class GpenKeyframeRestorer:
    # ONNX session 기반 GPEN 복원 런타임 묶음이다.
    session: "ort.InferenceSession"
    input_name: str
    input_size: int

    def __init__(self, *, session: "ort.InferenceSession", input_name: str, input_size: int) -> None:
        self.session = session
        self.input_name = input_name
        self.input_size = input_size


def _resolve_gpen_providers(provider: str) -> list[str | tuple[str, dict[str, str | int]]]:
    # CLI에서 쓰기 쉬운 provider 이름을 onnxruntime provider 설정으로 바꾼다.
    if provider == "cuda":
        return [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    if provider == "coreml":
        return [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                },
            ),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def load_gpen_keyframe_restorer(
    model_path: str,
    *,
    provider: str,
    input_size: int,
) -> GpenKeyframeRestorer:
    # 실행 전체에서 재사용할 GPEN session 하나를 만들어 둔다.
    if ort is None:
        raise RuntimeError("onnxruntime is required for GPEN keyframe restoration.")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=_resolve_gpen_providers(provider),
    )
    return GpenKeyframeRestorer(
        session=session,
        input_name=session.get_inputs()[0].name,
        input_size=int(input_size),
    )


def _preprocess_gpen(face_bgr: np.ndarray, input_size: int) -> np.ndarray:
    # GPEN은 [-1, 1] 범위의 정규화된 RGB, NCHW tensor를 기대한다.
    resized = cv2.resize(face_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = rgb / 255.0
    rgb = rgb * 2.0 - 1.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def _postprocess_gpen(output: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    # GPEN 출력을 이후 OpenCV 코드가 쓰는 일반 uint8 BGR로 되돌린다.
    image = output[0].transpose(1, 2, 0)
    image = ((image + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.resize(image, out_size, interpolation=cv2.INTER_LINEAR)


def restore_face_crop_with_gpen(face_bgr: np.ndarray, restorer: GpenKeyframeRestorer) -> np.ndarray:
    # 얼굴 crop 하나에 대해 GPEN 전처리/추론/후처리를 묶어서 수행하는 헬퍼다.
    blob = _preprocess_gpen(face_bgr, restorer.input_size)
    output = restorer.session.run(None, {restorer.input_name: blob})[0]
    return _postprocess_gpen(output, (face_bgr.shape[1], face_bgr.shape[0]))


def _expand_mask(mask_uint8: np.ndarray, expand_px: int) -> np.ndarray:
    # 유효 영역을 조금 넓혀 두면 복원된 영역과 아닌 영역이 만나는 경계 seam이 덜 보인다.
    if expand_px <= 0:
        return np.asarray(mask_uint8, dtype=np.uint8)
    kernel_size = max(1, int(expand_px) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(np.asarray(mask_uint8, dtype=np.uint8), kernel, iterations=1)


def restore_keyframe_face_region(
    face_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    *,
    gpen_restorer: GpenKeyframeRestorer | None,
    mask_expand_px: int,
    feather_px: int,
) -> np.ndarray:
    # warp된 keyframe crop에 GPEN 복원 모델을 적용한다.
    # 필요하면 warp coverage mask 안쪽 얼굴 영역으로만 효과를 제한할 수도 있다.
    if gpen_restorer is None:
        return face_bgr
    if mask_expand_px < 0:
        restored = np.asarray(face_bgr, dtype=np.uint8)
        restored = restore_face_crop_with_gpen(restored, gpen_restorer)
        return restored

    expanded_mask = _expand_mask(mask_uint8, int(mask_expand_px))
    ys, xs = np.where(expanded_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return face_bgr

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    source_roi = np.asarray(face_bgr[y1:y2, x1:x2], dtype=np.uint8)
    restored_roi = restore_face_crop_with_gpen(source_roi.copy(), gpen_restorer)

    alpha_mask = expanded_mask[y1:y2, x1:x2]
    if feather_px > 0:
        blur_size = max(1, int(feather_px) * 2 + 1)
        alpha_mask = cv2.GaussianBlur(alpha_mask, (blur_size, blur_size), 0)
    alpha = np.asarray(alpha_mask, dtype=np.float32) / 255.0

    blended_roi = (
        source_roi.astype(np.float32) * (1.0 - alpha[..., None]) +
        restored_roi.astype(np.float32) * alpha[..., None]
    )
    output = np.asarray(face_bgr, dtype=np.uint8).copy()
    output[y1:y2, x1:x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)
    return output
