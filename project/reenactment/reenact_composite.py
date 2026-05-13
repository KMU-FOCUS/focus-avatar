from __future__ import annotations

# reenact에서 쓰는 저수준 face warp / blend / composite 헬퍼 모음이다.
# 생성된 아바타 얼굴 crop을 목적지 영상 ROI와 시각적으로 합치는 역할을 하며,
# mask와 color matching도 여기서 처리한다.

from typing import Sequence

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError

import metadata_bbox_utils as overlay_helpers
from reenact_assets_runtime import CROP_SIZE


DEFAULT_CLONE_MODE = "alpha"
DEFAULT_ALPHA = 1.0
DEFAULT_FEATHER_PX = 24
DEFAULT_MASK_EXPAND_PX = 0
DEFAULT_MASK_GAMMA = 0.8
DEFAULT_COLOR_MATCH = "lab"
DEFAULT_COLOR_MATCH_STRENGTH = 0.45
DEFAULT_COLOR_MATCH_EXPAND_PX = 0
DEFAULT_COLOR_MATCH_DOWNSAMPLE = 1
DEFAULT_OVERRIDE_MASK_BLUR_KSIZE = 9


def _odd_kernel_size(radius_px: int) -> int:
    return max(1, int(radius_px) * 2 + 1)


def apply_affine(
    src: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
    output: np.ndarray,
    mask: np.ndarray,
) -> None:
    # source crop 공간의 삼각형 하나를 destination crop 공간으로 warp한다.
    # 얼굴 전체 warp는 이런 삼각형 단위 warp를 반복해서 만든다.
    # 여기서 중요한 점은 "이미지 전체를 한 번에 휘게 만드는 것"이 아니라
    # landmark 삼각형별로 잘라서 affine transform을 적용한다는 것이다.
    # 이렇게 해야 표정/포즈 변화가 있어도 국소적으로 더 자연스럽게 따라간다.
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    dst_rect = cv2.boundingRect(np.float32([dst_tri]))

    x1, y1, w1, h1 = src_rect
    x2, y2, w2, h2 = dst_rect
    if min(w1, h1, w2, h2) <= 0:
        return

    src_crop = src[y1 : y1 + h1, x1 : x1 + w1]
    src_local = np.float32([[p[0] - x1, p[1] - y1] for p in src_tri])
    dst_local = np.float32([[p[0] - x2, p[1] - y2] for p in dst_tri])

    # source triangle -> destination triangle로 가는 2D affine 변환을 구한다.
    # 이후 warpAffine은 source crop의 해당 삼각형 주변 픽셀을 dst_rect 크기로 변형한다.
    warp_mat = cv2.getAffineTransform(src_local, dst_local)
    warped = cv2.warpAffine(
        src_crop,
        warp_mat,
        (w2, h2),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # tri_mask는 "이번 삼각형이 차지하는 유효 영역"만 output에 반영하기 위한 마스크다.
    # 여러 삼각형 결과가 같은 output 배열에 누적되므로,
    # 삼각형 내부만 덮어쓰고 바깥은 유지해야 경계가 깨지지 않는다.
    tri_mask = np.zeros((h2, w2, 3), dtype=np.float32)
    cv2.fillConvexPoly(tri_mask, np.int32(dst_local), (1.0, 1.0, 1.0), 16, 0)

    # output에는 실제 warp된 픽셀을 누적하고,
    # mask에는 어떤 위치가 얼굴로 채워졌는지 coverage를 쌓아 둔다.
    # 이 coverage는 나중에 얼굴 경계 블렌딩과 face-only mask 계산의 기반이 된다.
    output[y2 : y2 + h2, x2 : x2 + w2] = (
        output[y2 : y2 + h2, x2 : x2 + w2] * (1.0 - tri_mask) + warped * tri_mask
    )
    mask[y2 : y2 + h2, x2 : x2 + w2] = np.maximum(mask[y2 : y2 + h2, x2 : x2 + w2], tri_mask)


def _fallback_warp_face(
    source_crop: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # 조각별 warp가 불가능한 경우에는 전체 crop에 대한 affine 근사로 한 번 더 시도한다.
    # 이것도 실패하면 합성만 건너뛸 수 있도록 빈 coverage mask를 반환한다.
    if len(src_points) < 3 or len(dst_points) < 3:
        return source_crop.copy(), np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)

    affine_mat, _ = cv2.estimateAffinePartial2D(
        np.asarray(src_points, dtype=np.float32),
        np.asarray(dst_points, dtype=np.float32),
        method=cv2.LMEDS,
    )
    if affine_mat is None:
        return source_crop.copy(), np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)

    warped_face = cv2.warpAffine(
        source_crop,
        affine_mat,
        (CROP_SIZE, CROP_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    full_mask = np.full((source_crop.shape[0], source_crop.shape[1]), 255, dtype=np.uint8)
    warped_mask = cv2.warpAffine(
        full_mask,
        affine_mat,
        (CROP_SIZE, CROP_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_face, warped_mask


def warp_face(
    source_crop: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # 조각별(piecewise-affine) 얼굴 warp 단계다.
    # 1) destination landmark로 삼각분할
    # 2) 대응하는 source 삼각형 warp
    # 3) 나중 blend에 쓸 coverage mask 누적
    # 랜드마크가 crop 경계를 약간 벗어나는 경우를 막기 위해 좌표를 clip한다.
    # clip이 없으면 Delaunay 삼각형이나 affine warp가 crop 바깥 좌표를 참조하면서
    # 경계 artifact나 OpenCV 오류를 만들 수 있다.
    src = np.clip(np.asarray(src_points, dtype=np.float32), 0, CROP_SIZE - 1)
    dst = np.clip(np.asarray(dst_points, dtype=np.float32), 0, CROP_SIZE - 1)
    try:
        tri = Delaunay(dst)
    except (QhullError, ValueError):
        return _fallback_warp_face(source_crop, src, dst)
    output = np.zeros_like(source_crop, dtype=np.float32)
    mask = np.zeros_like(source_crop, dtype=np.float32)

    # destination landmark 기준으로 삼각분할을 만든 뒤,
    # 각 삼각형에 대응하는 source 영역을 destination으로 끌어온다.
    # 전체 얼굴 warp는 결국 이 루프의 반복이다.
    for simplex in tri.simplices:
        apply_affine(source_crop, src[simplex], dst[simplex], output, mask)

    # output은 float 누적 결과라 마지막에 uint8 얼굴 이미지로 바꾸고,
    # mask는 3채널 coverage 중 최대값을 써서 단일 채널 얼굴 마스크로 정리한다.
    # 이 coverage_mask는 "warp로 실제 채워진 얼굴 영역"이라 후단 합성에서 유용하다.
    face_only = np.clip(output, 0, 255).astype(np.uint8)
    coverage_mask = np.clip(np.max(mask, axis=2) * 255.0, 0, 255).astype(np.uint8)
    return face_only, coverage_mask


def build_face_mask(
    points: np.ndarray,
    *,
    size: int = CROP_SIZE,
    feather_px: int = 24,
    expand_px: int = 0,
    center_gamma: float = 0.8,
) -> np.ndarray:
    # destination landmark hull로부터 부드러운 alpha mask를 만든다.
    # feather와 expand 값으로 이마/헤어라인을 얼마나 포함할지 조절한다.
    # 여기서 만들어지는 mask는 "어디를 얼굴로 볼 것인가"를 정의하는 핵심이다.
    # seam이 보이거나 얼굴 외곽이 딱 잘려 보이면 이 함수의 영향이 크다.
    mask = np.zeros((size, size), dtype=np.uint8)
    hull = cv2.convexHull(np.asarray(points, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    # expand_px는 convex hull을 바깥으로 조금 확장해
    # 헤어라인/턱선이 너무 타이트하게 잘리는 현상을 줄이는 용도다.
    # 너무 크게 잡으면 얼굴 밖 배경까지 섞여 경계가 번질 수 있다.
    if expand_px > 0:
        kernel_size = _odd_kernel_size(expand_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if feather_px <= 0:
        return mask

    # distanceTransform으로 경계에서 안쪽으로 갈수록 값이 커지는 지도를 만든다.
    # 이를 feather_px로 나누면 경계가 부드럽게 0->1로 올라가는 alpha가 된다.
    # center_gamma는 중앙부를 더 단단하게 유지할지, 전체적으로 부드럽게 만들지를 조절한다.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    alpha = np.clip(dist / float(feather_px), 0.0, 1.0)
    if center_gamma > 0 and abs(center_gamma - 1.0) > 1e-6:
        alpha = np.power(alpha, float(center_gamma))
    return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def _weighted_channel_stats(channel: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    # 가중 통계는 배경이 아니라 얼굴 픽셀 중심으로 color matching을 하게 해준다.
    # 즉 ROI 전체 평균이 아니라 "마스크가 큰 얼굴 영역" 위주로 평균/표준편차를 계산한다.
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-6:
        return float(np.mean(channel)), float(np.std(channel))
    mean = float(np.sum(channel * weights) / weight_sum)
    variance = float(np.sum(((channel - mean) ** 2) * weights) / weight_sum)
    return mean, max(variance ** 0.5, 1e-6)


def _expand_mask(mask_uint8: np.ndarray, expand_px: int) -> np.ndarray:
    # color matching 통계용 mask를 약간 넓힐 때 쓴다.
    # 블렌딩용 얼굴 mask와 stats용 mask를 완전히 같게 둘 필요는 없어서 helper로 분리했다.
    if expand_px <= 0:
        return mask_uint8
    kernel_size = _odd_kernel_size(expand_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask_uint8, kernel, iterations=1)


def _downsample_for_stats(
    image: np.ndarray,
    *,
    factor: int,
    interpolation: int,
) -> np.ndarray:
    # color 통계는 고해상도 전체에서 꼭 계산할 필요가 없어서 downsample 옵션을 둔다.
    # 큰 ROI에서 통계 계산 비용을 줄이기 위한 경량화 포인트다.
    if factor <= 1:
        return image
    h, w = image.shape[:2]
    out_w = max(1, int(round(w / float(factor))))
    out_h = max(1, int(round(h / float(factor))))
    return cv2.resize(image, (out_w, out_h), interpolation=interpolation)


def _apply_face_mask_override(
    base_mask: np.ndarray,
    face_mask_override: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    resized_face_mask = cv2.resize(
        np.asarray(face_mask_override, dtype=np.uint8),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_face_mask = cv2.GaussianBlur(
        resized_face_mask,
        (DEFAULT_OVERRIDE_MASK_BLUR_KSIZE, DEFAULT_OVERRIDE_MASK_BLUR_KSIZE),
        0,
    )
    return np.minimum(base_mask, resized_face_mask)


def _alpha_blend_roi(
    target_roi: np.ndarray,
    source_face: np.ndarray,
    mask_uint8: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    roi = target_roi.astype(np.float32)
    mask_alpha = (mask_uint8.astype(np.float32) / 255.0)[..., None] * float(alpha)
    blended = roi * (1.0 - mask_alpha) + source_face.astype(np.float32) * mask_alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def match_face_color(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    *,
    mode: str,
    strength: float,
    stats_expand_px: int = 0,
    stats_downsample: int = 1,
) -> np.ndarray:
    # 필요하면 avatar 색을 destination ROI의 통계 쪽으로 맞춘다.
    # 광원이나 색감이 다를 때 "스티커를 붙여 놓은 느낌"을 줄이기 위한 단계다.
    # 합성 경계가 튄다고 느껴질 때는 mask뿐 아니라 이 color matching 단계도 중요하다.
    if mode == "none" or strength <= 0.0:
        return source_bgr

    # 통계용 mask를 약간 확장하면 얼굴 외곽 인접부의 색감도 같이 반영할 수 있다.
    # 다만 과도하게 넓히면 배경색까지 끌려와 오히려 피부톤이 틀어질 수 있다.
    stats_mask_uint8 = _expand_mask(mask_uint8, int(stats_expand_px))

    # LAB는 밝기/색차를 어느 정도 분리해서 다룰 수 있어
    # 얼굴 톤 매칭에서 RGB보다 자연스럽게 보이는 경우가 많다.
    # mode="rgb"는 단순 채널 통계를 그대로 맞춘다.
    if mode == "rgb":
        source_space = source_bgr.astype(np.float32)
        target_space = target_bgr.astype(np.float32)
        to_bgr = None
    else:
        source_space = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_space = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        to_bgr = cv2.COLOR_LAB2BGR

    # 통계 계산용 downsample은 속도를 위한 것이고,
    # 실제 보정은 원본 해상도 source_space 전체에 적용된다.
    stats_factor = max(1, int(stats_downsample))
    stats_source_space = _downsample_for_stats(
        source_space,
        factor=stats_factor,
        interpolation=cv2.INTER_AREA if stats_factor > 1 else cv2.INTER_LINEAR,
    )
    stats_target_space = _downsample_for_stats(
        target_space,
        factor=stats_factor,
        interpolation=cv2.INTER_AREA if stats_factor > 1 else cv2.INTER_LINEAR,
    )
    stats_weights = _downsample_for_stats(
        stats_mask_uint8,
        factor=stats_factor,
        interpolation=cv2.INTER_AREA if stats_factor > 1 else cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0
    adjusted = source_space.copy()

    # 각 채널별로 source를 z-score 비슷하게 정규화한 뒤
    # target의 평균/표준편차 쪽으로 옮겨 준다.
    # 완전 치환은 과해 보일 수 있으므로 마지막에 strength로 원본과 섞는다.
    for channel_idx in range(3):
        src_channel = source_space[..., channel_idx]
        dst_channel = target_space[..., channel_idx]
        stats_src_channel = stats_source_space[..., channel_idx]
        stats_dst_channel = stats_target_space[..., channel_idx]
        src_mean, src_std = _weighted_channel_stats(stats_src_channel, stats_weights)
        dst_mean, dst_std = _weighted_channel_stats(stats_dst_channel, stats_weights)
        normalized = (src_channel - src_mean) * (dst_std / src_std) + dst_mean
        adjusted[..., channel_idx] = normalized

    adjusted = np.clip(adjusted, 0, 255)
    if to_bgr is not None:
        adjusted_bgr = cv2.cvtColor(adjusted.astype(np.uint8), to_bgr).astype(np.float32)
    else:
        adjusted_bgr = adjusted.astype(np.float32)

    source_float = source_bgr.astype(np.float32)
    blended = source_float * (1.0 - strength) + adjusted_bgr * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def composite_face(
    frame_bgr: np.ndarray,
    face_crop_bgr: np.ndarray,
    bbox_xyxy: Sequence[float],
    crop_points: np.ndarray,
    *,
    face_mask_override: np.ndarray | None,
    clone_mode: str = DEFAULT_CLONE_MODE,
    alpha: float = DEFAULT_ALPHA,
    feather_px: int = DEFAULT_FEATHER_PX,
    mask_expand_px: int = DEFAULT_MASK_EXPAND_PX,
    mask_gamma: float = DEFAULT_MASK_GAMMA,
    color_match: str = DEFAULT_COLOR_MATCH,
    color_match_strength: float = DEFAULT_COLOR_MATCH_STRENGTH,
    color_match_expand_px: int = DEFAULT_COLOR_MATCH_EXPAND_PX,
    color_match_downsample: int = DEFAULT_COLOR_MATCH_DOWNSAMPLE,
) -> np.ndarray:
    # warp된 얼굴 crop을 최종적으로 destination frame에 되돌려 붙이는 단계다.
    # face/mask를 ROI 크기로 맞추고, 필요하면 color match 후 blend한다.
    # 실제 seam 품질은 여기서 거의 결정된다.
    # 즉 "얼굴이 어디에 붙는지", "어떤 mask로 붙는지", "색을 얼마나 맞추는지",
    # "alpha로 섞을지 seamlessClone을 쓸지"가 모두 이 함수 안에 있다.
    clipped = overlay_helpers.clamp_box(
        tuple(int(round(v)) for v in bbox_xyxy[:4]),
        frame_bgr.shape[1],
        frame_bgr.shape[0],
    )
    if clipped is None:
        return frame_bgr

    x1, y1, x2, y2 = clipped
    w = x2 - x1
    h = y2 - y1

    # face_crop_bgr와 crop-space mask는 256x256 기준이므로
    # 실제 target bbox ROI 크기에 맞게 다시 resize해야 한다.
    resized_face = cv2.resize(face_crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(
        build_face_mask(
            crop_points,
            feather_px=feather_px,
            expand_px=mask_expand_px,
            center_gamma=mask_gamma,
        ),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    # warp_face가 반환한 coverage mask가 있으면
    # landmark hull 기반 soft mask와 교집합처럼 써서
    # "실제 warp가 채워진 영역만" 좀 더 보수적으로 합성하게 만든다.
    # 이 부분이 턱선/볼 외곽 seam에 영향을 많이 준다.
    if face_mask_override is not None:
        resized_mask = _apply_face_mask_override(
            resized_mask,
            face_mask_override,
            width=w,
            height=h,
        )
    target_roi = frame_bgr[y1:y2, x1:x2]

    # 색 보정을 먼저 적용한 뒤에 blending/clone을 수행한다.
    # 즉 seam을 줄이는 전략은
    # 1) mask를 좋게 만들고
    # 2) source와 target 색을 맞추고
    # 3) 마지막에 적절히 섞는 3단계라고 볼 수 있다.
    resized_face = match_face_color(
        resized_face,
        target_roi,
        resized_mask,
        mode=color_match,
        strength=float(color_match_strength),
        stats_expand_px=int(color_match_expand_px),
        stats_downsample=int(color_match_downsample),
    )

    # alpha blend 대신 seamlessClone 계열을 시도할 수도 있다.
    # 조명/텍스처 차이가 큰 경우 clone이 자연스러울 수 있지만,
    # 항상 더 좋은 것은 아니고 실패하면 그냥 alpha blend로 fallback한다.
    if clone_mode != "alpha":
        try:
            center = (x1 + w // 2, y1 + h // 2)
            frame_bgr = cv2.seamlessClone(
                resized_face,
                frame_bgr,
                resized_mask,
                center,
                cv2.NORMAL_CLONE if clone_mode == "normal" else cv2.MIXED_CLONE,
            )
            if alpha >= 0.999:
                return frame_bgr
        except Exception:
            pass

    # 기본 경로는 soft alpha blending이다.
    # resized_mask가 0에 가까운 곳은 원본 ROI가 유지되고,
    # 1에 가까운 곳은 reenacted face가 강하게 들어간다.
    # alpha 인자는 전체 합성 강도를 한 번 더 줄이거나 늘리는 global scale 역할을 한다.
    frame_bgr[y1:y2, x1:x2] = _alpha_blend_roi(
        frame_bgr[y1:y2, x1:x2],
        resized_face,
        resized_mask,
        alpha=alpha,
    )
    return frame_bgr
