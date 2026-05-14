from __future__ import annotations

# Qualcomm / FaceMap 3DMM coefficient를
# ARKit52 스타일 표정값과 복원 landmark로 바꿀 때 쓰는 핵심 수학 헬퍼 모듈이다.
# 상위 변환 스크립트와 reenact 스크립트 여러 개가 이 모듈에 의존한다.

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


ARKIT_52_BLENDSHAPES = (
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
)


IMAGE_LEFT_BROW = (17, 18, 19, 20, 21)
IMAGE_RIGHT_BROW = (22, 23, 24, 25, 26)
IMAGE_LEFT_EYE = (36, 37, 38, 39, 40, 41)
IMAGE_RIGHT_EYE = (42, 43, 44, 45, 46, 47)
IMAGE_LEFT_MOUTH_CORNER = 48
IMAGE_RIGHT_MOUTH_CORNER = 54

DEFAULT_BLENDSHAPE_DEADZONE = 0.03
DEFAULT_BLENDSHAPE_SMOOTHING = 0.35
DEFAULT_BLENDSHAPE_MAXS: Mapping[str, float] = {
    "eyeLookDownLeft": 0.65,
    "eyeLookDownRight": 0.65,
    "eyeLookInLeft": 0.65,
    "eyeLookInRight": 0.65,
    "eyeLookOutLeft": 0.65,
    "eyeLookOutRight": 0.65,
    "eyeLookUpLeft": 0.65,
    "eyeLookUpRight": 0.65,
    "eyeSquintLeft": 0.85,
    "eyeSquintRight": 0.85,
    "mouthRollLower": 0.70,
    "mouthRollUpper": 0.70,
    "mouthShrugLower": 0.75,
    "mouthShrugUpper": 0.75,
    "mouthUpperUpLeft": 0.75,
    "mouthUpperUpRight": 0.75,
    "cheekSquintLeft": 0.70,
    "cheekSquintRight": 0.70,
    "noseSneerLeft": 0.45,
    "noseSneerRight": 0.45,
    "tongueOut": 0.0,
}


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _distance(points: np.ndarray, a: int, b: int) -> float:
    return _safe_norm(points[a] - points[b])


def _mean_points(points: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return np.mean(points[np.asarray(indices, dtype=np.int32)], axis=0)


def _eye_openness(points: np.ndarray, eye_indices: Sequence[int]) -> float:
    a, b, c, d, e, f = eye_indices
    width = max(_distance(points, a, d), 1e-6)
    height = 0.5 * (_distance(points, b, f) + _distance(points, c, e))
    return height / width


def _to_numpy(vector: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


@dataclass(frozen=True)
class Qualcomm3DMMCoefficients:
    # 원시 264 길이 FaceMap 출력 벡터를 구조적으로 표현한 객체다.
    # 상위 코드에서는 매번 인덱스를 직접 자르기보다 이 형태를 선호한다.
    identity: np.ndarray
    expression: np.ndarray
    pitch: float
    yaw: float
    roll: float
    translation_x: float
    translation_y: float
    focal_length: float

    @property
    def output_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.identity.reshape(-1),
                self.expression.reshape(-1),
                np.asarray(
                    [
                        self.pitch,
                        self.yaw,
                        self.roll,
                        self.translation_x,
                        self.translation_y,
                        self.focal_length,
                    ],
                    dtype=np.float32,
                ),
            ]
        )


@dataclass(frozen=True)
class ReconstructedFace:
    # 3D 얼굴을 복원한 뒤 다시 이미지/crop 공간으로 사영한 결과를 담는 객체다.
    landmarks_2d: np.ndarray
    landmarks_3d: np.ndarray
    pose_radians: Mapping[str, float]
    camera: Mapping[str, float]


def split_qualcomm_3dmm_output(
    output: Sequence[float] | np.ndarray,
) -> Qualcomm3DMMCoefficients:
    """
    Qualcomm FaceMap 3DMM output layout.

    Official post-processing uses:
    - identity coefficients: 219
    - expression coefficients: 39
    - pose/translation/focal: 6

    Total length: 264.
    """

    vector = _to_numpy(output, name="output")
    if vector.size != 264:
        raise ValueError(f"Qualcomm FaceMap output must have length 264, got {vector.size}.")

    return Qualcomm3DMMCoefficients(
        identity=vector[0:219].copy(),
        expression=vector[219:258].copy(),
        pitch=float(vector[258]),
        yaw=float(vector[259]),
        roll=float(vector[260]),
        translation_x=float(vector[261]),
        translation_y=float(vector[262]),
        focal_length=float(vector[263]),
    )


def transform_crop_landmarks_to_image(
    landmarks_2d: np.ndarray,
    bbox_xyxy: Sequence[float],
    resized_height: int,
    resized_width: int,
) -> np.ndarray:
    """
    Match Qualcomm's official coordinate restoration from crop space to image space.
    """

    points = np.asarray(landmarks_2d, dtype=np.float32).reshape(-1, 2).copy()
    x0, y0, x1, y1 = map(float, bbox_xyxy)
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)

    points[:, 0] = (points[:, 0] + resized_width * 0.5) * width / resized_width + x0
    points[:, 1] = (points[:, 1] + resized_height * 0.5) * height / resized_height + y0
    return points


def reconstruct_qualcomm_68_landmarks(
    coefficients: Qualcomm3DMMCoefficients | Sequence[float] | np.ndarray,
    mean_face: np.ndarray,
    shape_basis: np.ndarray,
    blendshape_basis: np.ndarray,
) -> ReconstructedFace:
    """
    Reconstruct 68 3D landmarks following Qualcomm's published post-processing.

    Required asset shapes:
    - mean_face: (204,) or (68, 3)
    - shape_basis: (204, 219)
    - blendshape_basis: (204, 39)
    """

    # 호출부가 raw 264 벡터를 주든, 이미 분해된 dataclass를 주든
    # 같은 복원 경로를 타게 하기 위해 둘 다 허용한다.
    coeffs = (
        coefficients
        if isinstance(coefficients, Qualcomm3DMMCoefficients)
        else split_qualcomm_3dmm_output(coefficients)
    )

    face = np.asarray(mean_face, dtype=np.float32).reshape(68 * 3, 1)
    basis_id = np.asarray(shape_basis, dtype=np.float32).reshape(68 * 3, 219)
    basis_exp = np.asarray(blendshape_basis, dtype=np.float32).reshape(68 * 3, 39)

    alpha_id = (coeffs.identity * 3.0).reshape(219, 1)
    alpha_exp = (coeffs.expression * 0.5 + 0.5).reshape(39, 1)
    pitch = float(coeffs.pitch) * np.pi / 2.0
    yaw = float(coeffs.yaw) * np.pi / 2.0
    roll = float(coeffs.roll) * np.pi / 2.0
    tx = float(coeffs.translation_x) * 60.0
    ty = float(coeffs.translation_y) * 60.0
    tz = 500.0
    focal = float(coeffs.focal_length) * 150.0 + 450.0

    p_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(-np.pi), -np.sin(-np.pi)],
            [0.0, np.sin(-np.pi), np.cos(-np.pi)],
        ],
        dtype=np.float32,
    )
    roll_matrix = np.asarray(
        [
            [np.cos(-roll), -np.sin(-roll), 0.0],
            [np.sin(-roll), np.cos(-roll), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    yaw_matrix = np.asarray(
        [
            [np.cos(-yaw), 0.0, np.sin(-yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(-yaw), 0.0, np.cos(-yaw)],
        ],
        dtype=np.float32,
    )
    pitch_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(-pitch), -np.sin(-pitch)],
            [0.0, np.sin(-pitch), np.cos(-pitch)],
        ],
        dtype=np.float32,
    )
    rotation = yaw_matrix @ pitch_matrix @ p_matrix @ roll_matrix

    vertices = (
        face + basis_id @ alpha_id + basis_exp @ alpha_exp
    ).reshape(68, 3) @ rotation.T
    vertices[:, 0] += tx
    vertices[:, 1] += ty
    vertices[:, 2] += tz

    projected = vertices[:, :2] * focal / tz
    return ReconstructedFace(
        landmarks_2d=projected.astype(np.float32),
        landmarks_3d=vertices.astype(np.float32),
        pose_radians={"pitch": pitch, "yaw": yaw, "roll": roll},
        camera={"focal_length": focal, "translation_x": tx, "translation_y": ty, "translation_z": tz},
    )


class Qualcomm3DMMToARKit52Converter:
    """
    Heuristic converter from Qualcomm 3DMM outputs to ARKit-style 52 blendshapes.

    This is not a learned semantic mapping because Qualcomm does not publish a direct
    39-exp -> ARKit52 conversion table. Instead, the converter:
    1. reconstructs 68 landmarks from the official 3DMM basis when assets are available
    2. extracts geometry proxies
    3. estimates the 52 ARKit blendshape values in [0, 1]

    Shapes that are impossible to infer reliably from 68 landmarks alone, such as
    `tongueOut`, are kept near zero.
    """

    def __init__(
        self,
        *,
        mirror_input: bool = False,
        neutral_momentum: float = 0.90,
        use_head_pose_as_eye_gaze: bool = False,
        blendshape_deadzone: float = DEFAULT_BLENDSHAPE_DEADZONE,
        blendshape_smoothing: float = DEFAULT_BLENDSHAPE_SMOOTHING,
        blendshape_maxs: Mapping[str, float] | None = None,
    ) -> None:
        self.mirror_input = mirror_input
        self.neutral_momentum = float(np.clip(neutral_momentum, 0.0, 0.999))
        self.use_head_pose_as_eye_gaze = use_head_pose_as_eye_gaze
        self.blendshape_deadzone = float(np.clip(blendshape_deadzone, 0.0, 0.25))
        self.blendshape_smoothing = float(np.clip(blendshape_smoothing, 0.0, 0.95))
        self.blendshape_maxs = dict(DEFAULT_BLENDSHAPE_MAXS)
        if blendshape_maxs:
            self.blendshape_maxs.update(
                {str(name): float(np.clip(value, 0.0, 1.0)) for name, value in blendshape_maxs.items()}
            )
        self._neutral_metrics: dict[str, float] | None = None
        self._neutral_expression: np.ndarray | None = None
        self._prev_shapes: dict[str, float] | None = None

    def reset(self) -> None:
        self._neutral_metrics = None
        self._neutral_expression = None
        self._prev_shapes = None

    def estimate_from_output(
        self,
        output: Sequence[float] | np.ndarray,
        *,
        mean_face: np.ndarray,
        shape_basis: np.ndarray,
        blendshape_basis: np.ndarray,
        update_neutral: bool = True,
    ) -> dict[str, float]:
        coeffs = split_qualcomm_3dmm_output(output)
        reconstructed = reconstruct_qualcomm_68_landmarks(
            coeffs,
            mean_face=mean_face,
            shape_basis=shape_basis,
            blendshape_basis=blendshape_basis,
        )
        return self.estimate_from_landmarks(
            reconstructed.landmarks_2d,
            expression_coeffs=coeffs.expression,
            pose_radians=reconstructed.pose_radians,
            update_neutral=update_neutral,
        )

    def estimate_from_landmarks(
        self,
        landmarks_2d: np.ndarray,
        *,
        expression_coeffs: Sequence[float] | np.ndarray | None = None,
        pose_radians: Mapping[str, float] | None = None,
        update_neutral: bool = True,
    ) -> dict[str, float]:
        points = np.asarray(landmarks_2d, dtype=np.float32).reshape(68, 2)
        metrics = self._extract_metrics(points)
        neutral = self._neutral_metrics or dict(metrics)
        shapes = self._metrics_to_blendshapes(metrics, neutral, pose_radians)
        shapes = self._postprocess_shapes(shapes)

        if update_neutral and self._should_update_neutral(shapes, pose_radians):
            self._update_neutral(metrics, expression_coeffs)

        return {name: _clamp01(shapes.get(name, 0.0)) for name in ARKIT_52_BLENDSHAPES}

    def _postprocess_shapes(self, shapes: Mapping[str, float]) -> dict[str, float]:
        processed = {name: _clamp01(float(shapes.get(name, 0.0))) for name in ARKIT_52_BLENDSHAPES}

        for name, max_value in self.blendshape_maxs.items():
            if name in processed:
                processed[name] = min(processed[name], max_value)

        if self.blendshape_deadzone > 0.0:
            for name, value in list(processed.items()):
                if value < self.blendshape_deadzone:
                    processed[name] = 0.0

        # Suppress contradictory combinations that tend to make faces look unstable.
        processed["eyeWideLeft"] *= 1.0 - 0.75 * max(processed["eyeBlinkLeft"], processed["eyeSquintLeft"])
        processed["eyeWideRight"] *= 1.0 - 0.75 * max(processed["eyeBlinkRight"], processed["eyeSquintRight"])

        left_smile_frown_overlap = min(processed["mouthSmileLeft"], processed["mouthFrownLeft"])
        right_smile_frown_overlap = min(processed["mouthSmileRight"], processed["mouthFrownRight"])
        processed["mouthSmileLeft"] -= left_smile_frown_overlap
        processed["mouthFrownLeft"] -= left_smile_frown_overlap
        processed["mouthSmileRight"] -= right_smile_frown_overlap
        processed["mouthFrownRight"] -= right_smile_frown_overlap

        processed["jawLeft"] = max(processed["jawLeft"] - 0.35 * processed["jawRight"], 0.0)
        processed["jawRight"] = max(processed["jawRight"] - 0.35 * processed["jawLeft"], 0.0)
        processed["mouthLeft"] = max(processed["mouthLeft"] - 0.35 * processed["mouthRight"], 0.0)
        processed["mouthRight"] = max(processed["mouthRight"] - 0.35 * processed["mouthLeft"], 0.0)

        if self.blendshape_smoothing > 0.0 and self._prev_shapes is not None:
            keep = self.blendshape_smoothing
            processed = {
                name: _clamp01(keep * float(self._prev_shapes.get(name, 0.0)) + (1.0 - keep) * value)
                for name, value in processed.items()
            }

        self._prev_shapes = dict(processed)
        return processed

    def _resolve_sides(self) -> dict[str, object]:
        if self.mirror_input:
            left_eye = IMAGE_LEFT_EYE
            right_eye = IMAGE_RIGHT_EYE
            left_brow = IMAGE_LEFT_BROW
            right_brow = IMAGE_RIGHT_BROW
            left_corner = IMAGE_LEFT_MOUTH_CORNER
            right_corner = IMAGE_RIGHT_MOUTH_CORNER
            left_nose = 31
            right_nose = 35
            left_upper_outer = 49
            right_upper_outer = 53
            left_lower_outer = 59
            right_lower_outer = 55
        else:
            left_eye = IMAGE_RIGHT_EYE
            right_eye = IMAGE_LEFT_EYE
            left_brow = IMAGE_RIGHT_BROW
            right_brow = IMAGE_LEFT_BROW
            left_corner = IMAGE_RIGHT_MOUTH_CORNER
            right_corner = IMAGE_LEFT_MOUTH_CORNER
            left_nose = 35
            right_nose = 31
            left_upper_outer = 53
            right_upper_outer = 49
            left_lower_outer = 55
            right_lower_outer = 59

        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "left_brow": left_brow,
            "right_brow": right_brow,
            "left_corner": left_corner,
            "right_corner": right_corner,
            "left_nose": left_nose,
            "right_nose": right_nose,
            "left_upper_outer": left_upper_outer,
            "right_upper_outer": right_upper_outer,
            "left_lower_outer": left_lower_outer,
            "right_lower_outer": right_lower_outer,
        }

    def _extract_metrics(self, points: np.ndarray) -> dict[str, float]:
        side = self._resolve_sides()
        face_width = max(_distance(points, 0, 16), 1e-6)
        face_height = max(abs(float(points[8, 1] - points[27, 1])), face_width * 0.6, 1e-6)
        mouth_center = 0.5 * (points[51] + points[57])
        mouth_inner_open = (
            _distance(points, 61, 67) + _distance(points, 62, 66) + _distance(points, 63, 65)
        ) / (3.0 * face_height)
        mouth_outer_open = (
            _distance(points, 50, 58) + _distance(points, 51, 57) + _distance(points, 52, 56)
        ) / (3.0 * face_height)
        mouth_width = _distance(points, 48, 54) / face_width
        left_eye_center = _mean_points(points, side["left_eye"])
        right_eye_center = _mean_points(points, side["right_eye"])
        left_brow = side["left_brow"]
        right_brow = side["right_brow"]
        left_corner = int(side["left_corner"])
        right_corner = int(side["right_corner"])

        metrics = {
            "face_width": face_width,
            "face_height": face_height,
            "left_eye_open": _eye_openness(points, side["left_eye"]),
            "right_eye_open": _eye_openness(points, side["right_eye"]),
            "mouth_open_inner": mouth_inner_open,
            "mouth_open_outer": mouth_outer_open,
            "mouth_width": mouth_width,
            "mouth_center_x": float(mouth_center[0]),
            "nose_x": float(points[33, 0]),
            "chin_x": float(points[8, 0]),
            "chin_drop": float(points[8, 1] - points[33, 1]) / face_height,
            "mouth_center_y": float(mouth_center[1]),
            "upper_lip_center_y": float(points[51, 1]),
            "lower_lip_center_y": float(points[57, 1]),
            "upper_lip_thickness": _distance(points, 51, 62) / face_height,
            "lower_lip_thickness": _distance(points, 57, 66) / face_height,
            "left_corner_raise": float(mouth_center[1] - points[left_corner, 1]) / face_height,
            "right_corner_raise": float(mouth_center[1] - points[right_corner, 1]) / face_height,
            "left_corner_stretch": float(abs(points[left_corner, 0] - mouth_center[0])) / face_width,
            "right_corner_stretch": float(abs(points[right_corner, 0] - mouth_center[0])) / face_width,
            "left_outer_brow_gap": float(left_eye_center[1] - points[left_brow[0], 1]) / face_height,
            "left_inner_brow_gap": float(left_eye_center[1] - points[left_brow[-1], 1]) / face_height,
            "right_inner_brow_gap": float(right_eye_center[1] - points[right_brow[0], 1]) / face_height,
            "right_outer_brow_gap": float(right_eye_center[1] - points[right_brow[-1], 1]) / face_height,
            "left_upper_nose_gap": float(points[int(side["left_upper_outer"]), 1] - points[int(side["left_nose"]), 1])
            / face_height,
            "right_upper_nose_gap": float(points[int(side["right_upper_outer"]), 1] - points[int(side["right_nose"]), 1])
            / face_height,
            "left_lower_chin_gap": float(points[8, 1] - points[int(side["left_lower_outer"]), 1]) / face_height,
            "right_lower_chin_gap": float(points[8, 1] - points[int(side["right_lower_outer"]), 1]) / face_height,
            "left_mouth_press_gap": _distance(points, int(side["left_upper_outer"]), 67 if self.mirror_input else 65)
            / face_height,
            "right_mouth_press_gap": _distance(points, int(side["right_upper_outer"]), 65 if self.mirror_input else 67)
            / face_height,
        }
        metrics["inner_brow_gap"] = 0.5 * (
            metrics["left_inner_brow_gap"] + metrics["right_inner_brow_gap"]
        )
        return metrics

    def _delta(self, metrics: Mapping[str, float], neutral: Mapping[str, float], key: str, scale: float) -> float:
        return (float(metrics[key]) - float(neutral[key])) / max(scale, 1e-6)

    def _metrics_to_blendshapes(
        self,
        metrics: Mapping[str, float],
        neutral: Mapping[str, float],
        pose_radians: Mapping[str, float] | None,
    ) -> dict[str, float]:
        face_height = float(metrics["face_height"])
        mouth_open_abs = _clamp01((float(metrics["mouth_open_inner"]) - 0.010) / 0.070)
        mouth_width_abs = _clamp01((float(metrics["mouth_width"]) - 0.34) / 0.22)
        left_blink_abs = _clamp01((0.30 - float(metrics["left_eye_open"])) / 0.18)
        right_blink_abs = _clamp01((0.30 - float(metrics["right_eye_open"])) / 0.18)
        left_wide_abs = _clamp01((float(metrics["left_eye_open"]) - 0.32) / 0.10)
        right_wide_abs = _clamp01((float(metrics["right_eye_open"]) - 0.32) / 0.10)

        left_blink = max(
            left_blink_abs,
            _clamp01(-self._delta(metrics, neutral, "left_eye_open", 0.08)),
        )
        right_blink = max(
            right_blink_abs,
            _clamp01(-self._delta(metrics, neutral, "right_eye_open", 0.08)),
        )
        jaw_open = max(
            mouth_open_abs,
            _clamp01(self._delta(metrics, neutral, "mouth_open_inner", 0.060)),
            _clamp01(self._delta(metrics, neutral, "chin_drop", 0.18)),
        )
        mouth_smile_left = max(
            _clamp01((float(metrics["left_corner_raise"]) - 0.010) / 0.080),
            _clamp01(self._delta(metrics, neutral, "left_corner_raise", 0.050)),
        )
        mouth_smile_right = max(
            _clamp01((float(metrics["right_corner_raise"]) - 0.010) / 0.080),
            _clamp01(self._delta(metrics, neutral, "right_corner_raise", 0.050)),
        )
        mouth_frown_left = max(
            _clamp01((-float(metrics["left_corner_raise"]) - 0.005) / 0.080),
            _clamp01(-self._delta(metrics, neutral, "left_corner_raise", 0.050)),
        )
        mouth_frown_right = max(
            _clamp01((-float(metrics["right_corner_raise"]) - 0.005) / 0.080),
            _clamp01(-self._delta(metrics, neutral, "right_corner_raise", 0.050)),
        )
        left_outer_up = max(
            _clamp01((float(metrics["left_outer_brow_gap"]) - 0.09) / 0.08),
            _clamp01(self._delta(metrics, neutral, "left_outer_brow_gap", 0.05)),
        )
        right_outer_up = max(
            _clamp01((float(metrics["right_outer_brow_gap"]) - 0.09) / 0.08),
            _clamp01(self._delta(metrics, neutral, "right_outer_brow_gap", 0.05)),
        )
        brow_inner_up = max(
            _clamp01((float(metrics["inner_brow_gap"]) - 0.095) / 0.08),
            _clamp01(self._delta(metrics, neutral, "inner_brow_gap", 0.05)),
        )
        brow_down_left = max(
            _clamp01((0.085 - float(metrics["left_inner_brow_gap"])) / 0.060),
            _clamp01(-self._delta(metrics, neutral, "left_inner_brow_gap", 0.04)),
        ) * (1.0 - 0.35 * left_blink)
        brow_down_right = max(
            _clamp01((0.085 - float(metrics["right_inner_brow_gap"])) / 0.060),
            _clamp01(-self._delta(metrics, neutral, "right_inner_brow_gap", 0.04)),
        ) * (1.0 - 0.35 * right_blink)

        mouth_pucker = max(
            _clamp01((0.43 - float(metrics["mouth_width"])) / 0.18) * _clamp01((0.050 - jaw_open) / 0.050),
            _clamp01(-self._delta(metrics, neutral, "mouth_width", 0.12)),
        )
        mouth_funnel = (
            _clamp01((0.48 - float(metrics["mouth_width"])) / 0.20)
            * _clamp01((float(metrics["mouth_open_outer"]) - 0.02) / 0.07)
        )
        mouth_stretch_left = max(
            _clamp01((float(metrics["left_corner_stretch"]) - 0.17) / 0.10),
            _clamp01(self._delta(metrics, neutral, "left_corner_stretch", 0.06)),
        )
        mouth_stretch_right = max(
            _clamp01((float(metrics["right_corner_stretch"]) - 0.17) / 0.10),
            _clamp01(self._delta(metrics, neutral, "right_corner_stretch", 0.06)),
        )
        mouth_center_shift = float(metrics["mouth_center_x"] - metrics["nose_x"]) / max(float(metrics["face_width"]), 1e-6)
        jaw_shift = mouth_center_shift + 0.55 * (
            float(metrics["chin_x"] - metrics["nose_x"]) / max(float(metrics["face_width"]), 1e-6)
        )

        if pose_radians:
            jaw_shift -= 0.12 * float(pose_radians.get("yaw", 0.0))

        upper_up_left = max(
            _clamp01((0.19 - float(metrics["left_upper_nose_gap"])) / 0.10),
            _clamp01(-self._delta(metrics, neutral, "left_upper_nose_gap", 0.05)),
        )
        upper_up_right = max(
            _clamp01((0.19 - float(metrics["right_upper_nose_gap"])) / 0.10),
            _clamp01(-self._delta(metrics, neutral, "right_upper_nose_gap", 0.05)),
        )
        lower_down_left = max(
            jaw_open * 0.55 + mouth_frown_left * 0.25,
            _clamp01(-self._delta(metrics, neutral, "left_lower_chin_gap", 0.07)),
        )
        lower_down_right = max(
            jaw_open * 0.55 + mouth_frown_right * 0.25,
            _clamp01(-self._delta(metrics, neutral, "right_lower_chin_gap", 0.07)),
        )
        mouth_press_left = max(
            _clamp01((0.045 - float(metrics["left_mouth_press_gap"])) / 0.03) * _clamp01((0.030 - jaw_open) / 0.030),
            _clamp01(-self._delta(metrics, neutral, "left_mouth_press_gap", 0.02)),
        )
        mouth_press_right = max(
            _clamp01((0.045 - float(metrics["right_mouth_press_gap"])) / 0.03) * _clamp01((0.030 - jaw_open) / 0.030),
            _clamp01(-self._delta(metrics, neutral, "right_mouth_press_gap", 0.02)),
        )

        chin_drop_delta = _clamp01(self._delta(metrics, neutral, "chin_drop", 0.12))
        mouth_close = _clamp01((chin_drop_delta - jaw_open) * 1.6)
        mouth_roll_upper = max(
            _clamp01((0.040 - float(metrics["upper_lip_thickness"])) / 0.025),
            _clamp01(-self._delta(metrics, neutral, "upper_lip_thickness", 0.018)),
        ) * _clamp01((0.030 - float(metrics["mouth_open_inner"])) / 0.030)
        mouth_roll_lower = max(
            _clamp01((0.040 - float(metrics["lower_lip_thickness"])) / 0.025),
            _clamp01(-self._delta(metrics, neutral, "lower_lip_thickness", 0.018)),
        ) * _clamp01((0.030 - float(metrics["mouth_open_inner"])) / 0.030)
        mouth_shrug_upper = _clamp01(upper_up_left * 0.5 + upper_up_right * 0.5 + mouth_close * 0.2)
        mouth_shrug_lower = _clamp01(
            max(
                _clamp01((float(metrics["left_lower_chin_gap"]) - 0.24) / 0.12),
                _clamp01((float(metrics["right_lower_chin_gap"]) - 0.24) / 0.12),
            )
            * 0.7
            + mouth_close * 0.2
        )
        cheek_puff = _clamp01(mouth_pucker * 0.7 * (1.0 - jaw_open))
        cheek_squint_left = _clamp01(0.5 * mouth_smile_left + 0.35 * (1.0 - left_wide_abs) + 0.2 * left_blink)
        cheek_squint_right = _clamp01(
            0.5 * mouth_smile_right + 0.35 * (1.0 - right_wide_abs) + 0.2 * right_blink
        )
        nose_sneer_left = _clamp01(0.55 * upper_up_left + 0.25 * mouth_smile_left + 0.15 * cheek_squint_left)
        nose_sneer_right = _clamp01(0.55 * upper_up_right + 0.25 * mouth_smile_right + 0.15 * cheek_squint_right)

        shapes = {name: 0.0 for name in ARKIT_52_BLENDSHAPES}
        shapes["browDownLeft"] = brow_down_left
        shapes["browDownRight"] = brow_down_right
        shapes["browInnerUp"] = brow_inner_up
        shapes["browOuterUpLeft"] = left_outer_up
        shapes["browOuterUpRight"] = right_outer_up
        shapes["cheekPuff"] = cheek_puff
        shapes["cheekSquintLeft"] = cheek_squint_left
        shapes["cheekSquintRight"] = cheek_squint_right
        shapes["eyeBlinkLeft"] = left_blink
        shapes["eyeBlinkRight"] = right_blink
        shapes["eyeSquintLeft"] = _clamp01(left_blink * 0.65 + cheek_squint_left * 0.25)
        shapes["eyeSquintRight"] = _clamp01(right_blink * 0.65 + cheek_squint_right * 0.25)
        shapes["eyeWideLeft"] = max(left_wide_abs, _clamp01(self._delta(metrics, neutral, "left_eye_open", 0.08)))
        shapes["eyeWideRight"] = max(
            right_wide_abs,
            _clamp01(self._delta(metrics, neutral, "right_eye_open", 0.08)),
        )
        shapes["jawForward"] = _clamp01(max(mouth_pucker, mouth_funnel) * 0.25)
        shapes["jawLeft"] = _clamp01(max(-jaw_shift, 0.0) / 0.06)
        shapes["jawOpen"] = jaw_open
        shapes["jawRight"] = _clamp01(max(jaw_shift, 0.0) / 0.06)
        shapes["mouthClose"] = mouth_close
        shapes["mouthDimpleLeft"] = _clamp01(mouth_smile_left * 0.55 + mouth_stretch_left * 0.25)
        shapes["mouthDimpleRight"] = _clamp01(mouth_smile_right * 0.55 + mouth_stretch_right * 0.25)
        shapes["mouthFrownLeft"] = mouth_frown_left
        shapes["mouthFrownRight"] = mouth_frown_right
        shapes["mouthFunnel"] = mouth_funnel
        shapes["mouthLeft"] = _clamp01(max(-mouth_center_shift, 0.0) / 0.05)
        shapes["mouthLowerDownLeft"] = _clamp01(lower_down_left)
        shapes["mouthLowerDownRight"] = _clamp01(lower_down_right)
        shapes["mouthPressLeft"] = mouth_press_left
        shapes["mouthPressRight"] = mouth_press_right
        shapes["mouthPucker"] = mouth_pucker
        shapes["mouthRight"] = _clamp01(max(mouth_center_shift, 0.0) / 0.05)
        shapes["mouthRollLower"] = mouth_roll_lower
        shapes["mouthRollUpper"] = mouth_roll_upper
        shapes["mouthShrugLower"] = mouth_shrug_lower
        shapes["mouthShrugUpper"] = mouth_shrug_upper
        shapes["mouthSmileLeft"] = mouth_smile_left
        shapes["mouthSmileRight"] = mouth_smile_right
        shapes["mouthStretchLeft"] = max(mouth_stretch_left, mouth_width_abs * 0.55)
        shapes["mouthStretchRight"] = max(mouth_stretch_right, mouth_width_abs * 0.55)
        shapes["mouthUpperUpLeft"] = upper_up_left
        shapes["mouthUpperUpRight"] = upper_up_right
        shapes["noseSneerLeft"] = nose_sneer_left
        shapes["noseSneerRight"] = nose_sneer_right

        if self.use_head_pose_as_eye_gaze and pose_radians:
            yaw = float(pose_radians.get("yaw", 0.0))
            pitch = float(pose_radians.get("pitch", 0.0))
            left_amount = _clamp01(max(-yaw, 0.0) / 0.35)
            right_amount = _clamp01(max(yaw, 0.0) / 0.35)
            up_amount = _clamp01(max(-pitch, 0.0) / 0.25)
            down_amount = _clamp01(max(pitch, 0.0) / 0.25)
            shapes["eyeLookInLeft"] = left_amount
            shapes["eyeLookOutLeft"] = right_amount
            shapes["eyeLookInRight"] = right_amount
            shapes["eyeLookOutRight"] = left_amount
            shapes["eyeLookUpLeft"] = up_amount
            shapes["eyeLookUpRight"] = up_amount
            shapes["eyeLookDownLeft"] = down_amount
            shapes["eyeLookDownRight"] = down_amount

        return shapes

    def _should_update_neutral(
        self,
        shapes: Mapping[str, float],
        pose_radians: Mapping[str, float] | None,
    ) -> bool:
        yaw = abs(float((pose_radians or {}).get("yaw", 0.0)))
        pitch = abs(float((pose_radians or {}).get("pitch", 0.0)))
        roll = abs(float((pose_radians or {}).get("roll", 0.0)))
        if yaw > 0.45 or pitch > 0.35 or roll > 0.45:
            return False

        active = max(
            float(shapes["jawOpen"]),
            float(shapes["mouthSmileLeft"]),
            float(shapes["mouthSmileRight"]),
            float(shapes["mouthFrownLeft"]),
            float(shapes["mouthFrownRight"]),
            float(shapes["eyeBlinkLeft"]),
            float(shapes["eyeBlinkRight"]),
            float(shapes["browInnerUp"]),
            float(shapes["browDownLeft"]),
            float(shapes["browDownRight"]),
        )
        return active < 0.28

    def _update_neutral(
        self,
        metrics: Mapping[str, float],
        expression_coeffs: Sequence[float] | np.ndarray | None,
    ) -> None:
        if self._neutral_metrics is None:
            self._neutral_metrics = {key: float(value) for key, value in metrics.items()}
        else:
            keep = self.neutral_momentum
            self._neutral_metrics = {
                key: keep * float(self._neutral_metrics[key]) + (1.0 - keep) * float(value)
                for key, value in metrics.items()
            }

        if expression_coeffs is None:
            return

        expr = _to_numpy(expression_coeffs, name="expression_coeffs")
        if self._neutral_expression is None:
            self._neutral_expression = expr.copy()
            return

        keep = self.neutral_momentum
        self._neutral_expression = keep * self._neutral_expression + (1.0 - keep) * expr


def estimate_arkit52_from_qualcomm_output(
    output: Sequence[float] | np.ndarray,
    *,
    mean_face: np.ndarray,
    shape_basis: np.ndarray,
    blendshape_basis: np.ndarray,
    mirror_input: bool = False,
    use_head_pose_as_eye_gaze: bool = False,
) -> dict[str, float]:
    converter = Qualcomm3DMMToARKit52Converter(
        mirror_input=mirror_input,
        use_head_pose_as_eye_gaze=use_head_pose_as_eye_gaze,
    )
    return converter.estimate_from_output(
        output,
        mean_face=mean_face,
        shape_basis=shape_basis,
        blendshape_basis=blendshape_basis,
    )


__all__ = [
    "ARKIT_52_BLENDSHAPES",
    "Qualcomm3DMMCoefficients",
    "Qualcomm3DMMToARKit52Converter",
    "ReconstructedFace",
    "estimate_arkit52_from_qualcomm_output",
    "reconstruct_qualcomm_68_landmarks",
    "split_qualcomm_3dmm_output",
    "transform_crop_landmarks_to_image",
]