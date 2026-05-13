from __future__ import annotations

# reenact에서 재사용할 avatar/source bank를 만드는 스크립트다.
# 각 view(front/left/right)마다 다음 정보를 함께 저장한다.
# - 원본 인물 이미지
# - 미리 계산한 source crop / landmark / coeff
# - 합성용 soft face mask
# - 어떤 yaw 구간에서 이 view를 써야 하는지에 대한 profile 정보

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .precompute_source_avatar_assets import build_source_asset, save_source_asset

PROFILE_FILENAME = "profile.json"
DEFAULT_SOURCE_PAD_RATIO = 0.35
DEFAULT_LANDMARKER_TASK: str | None = None
DEFAULT_MEAN_FACE_PATH: str | None = None
DEFAULT_SHAPE_BASIS_PATH: str | None = None
DEFAULT_BLENDSHAPE_BASIS_PATH: str | None = None
DEFAULT_MASK_FEATHER_PX = 24
DEFAULT_MASK_EXPAND_PX = 8
DEFAULT_FRONT_YAW_RANGE = (-25.0, 25.0)
DEFAULT_LEFT_YAW_MIN = 25.0
DEFAULT_RIGHT_YAW_MAX = -25.0
DEFAULT_VIEW_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def parse_args() -> argparse.Namespace:
    # bank 생성은 렌더링마다 반복하는 작업이 아니라 사전 준비 단계다.
    # 여기서 만든 폴더는 이후 여러 reenact 실행에서 계속 재사용하는 것이 목적이다.
    # 그래서 입력도 "영상 한 번 돌릴 때 필요한 옵션"보다
    # "bank를 어떤 규칙으로 만들어 둘지"에 집중되어 있다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def copy_source_image_to_bank(image_path: str | Path, output_path: Path) -> None:
    # source 이미지를 bank 폴더 안의 PNG 파일로 정규화해 둔다.
    # 이후 profile.json은 상대 파일명만 들고 있게 하므로,
    # 원본 이미지가 어디에 있었는지와 무관하게 bank 폴더만 복사해도 재사용할 수 있게 된다.
    image = cv2.imread(str(Path(image_path).expanduser().resolve()))
    if image is None:
        raise RuntimeError(f"Failed to read source portrait: {image_path}")
    cv2.imwrite(str(output_path), image)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    # profile/manifest 모두 같은 포맷으로 저장하므로 JSON 쓰기 규칙을 한곳에 둔다.
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_source_dir(source_dir: str | Path) -> Path:
    # source-dir 하나만 넘기면 그 안에서 front/left/right 이미지를 찾는 구조다.
    resolved = Path(source_dir).expanduser().resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"Source dir was not found: {resolved}")
    return resolved


def resolve_bank_id(source_dir: Path) -> str:
    # 별도 bank-id를 받지 않고 source 폴더 이름을 그대로 bank id로 사용한다.
    return source_dir.name


def find_view_image_path(source_dir: Path, view_name: str) -> str | None:
    # front/left/right 고정 이름 규칙으로 이미지를 찾는다.
    # 여러 확장자를 허용하되, 같은 view에 여러 파일이 있으면 가장 먼저 찾은 하나만 쓴다.
    for suffix in DEFAULT_VIEW_IMAGE_SUFFIXES:
        candidate = source_dir / f"{view_name}{suffix}"
        if candidate.exists():
            return str(candidate)
    return None


def build_soft_face_mask(
    points: np.ndarray,
    *,
    size: int,
    feather_px: int,
    expand_px: int,
) -> np.ndarray:
    # source crop 기준의 얼굴 영역 mask를 미리 만들어 둔다.
    # reenact 때 이 mask를 사용하면 머리카락/이마가 포함되는 범위를
    # view별로 더 안정적으로 맞출 수 있다.
    # 절차는 다음과 같다.
    # 1. source landmark의 convex hull을 얼굴 대략 영역으로 본다.
    # 2. 필요하면 mask를 조금 확장해서 이마/턱 바깥 여유를 준다.
    # 3. 마지막으로 feathering 해서 가장자리가 부드럽게 섞이게 한다.
    # 이렇게 미리 저장해 두면 런타임 reenact에서는 view mask를 다시 만들 필요가 없다.
    mask = np.zeros((size, size), dtype=np.uint8)
    hull = cv2.convexHull(np.asarray(points, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    if expand_px > 0:
        kernel_size = max(1, int(expand_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if feather_px <= 0:
        return mask

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    alpha = np.clip(dist / float(feather_px), 0.0, 1.0)
    return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def build_view_entry(
    *,
    view_name: str,
    source_image_path: str | Path,
    bank_dir: Path,
) -> dict[str, Any]:
    # 한 개의 source view에 대해 재사용 가능한 산출물을 모두 만든다.
    # - 정규화된 source 이미지
    # - precompute된 .npz
    # - soft mask
    # - 미리보기/점검용 메타데이터
    # 즉 bank 안의 "front 하나", "left 하나" 같은 view 단위를 완성하는 함수다.
    # profile.json에는 최종적으로 이 함수가 만든 파일 이름들과 요약 정보가 들어간다.
    source_png_path = bank_dir / f"{view_name}.png"
    meta_path = bank_dir / f"{view_name}_meta.npz"
    mask_path = bank_dir / f"{view_name}_mask.png"

    # 원본 이미지를 bank 폴더 안에 고정된 이름으로 복사해 두면
    # 이후 profile.json 에서 각 view를 단순하게 참조할 수 있다.
    copy_source_image_to_bank(source_image_path, source_png_path)
    # source asset precompute는 별도 스크립트/모듈에 위임한다.
    # 여기서는 bank 생성기답게 "어디에 저장할지"와 "view별 결과를 어떻게 묶을지"만 관리한다.
    asset = build_source_asset(
        source_image_path=source_png_path,
        mean_face_path=DEFAULT_MEAN_FACE_PATH,
        shape_basis_path=DEFAULT_SHAPE_BASIS_PATH,
        blendshape_basis_path=DEFAULT_BLENDSHAPE_BASIS_PATH,
        source_pad_ratio=float(DEFAULT_SOURCE_PAD_RATIO),
        landmarker_task=DEFAULT_LANDMARKER_TASK,
    )
    save_source_asset(
        asset=asset,
        output_path=meta_path,
    )

    # 합성에서 바로 쓸 수 있도록 각 source view의 soft mask도 저장한다.
    # meta.npz에는 crop/landmark/coeff를 넣고, mask는 별도 png로 관리한다.
    # mask를 npz 안에 넣지 않고 분리해 두면 사람이 시각적으로 점검하기도 쉽고,
    # 나중에 mask만 따로 교체하거나 재생성하기도 편하다.
    mask = build_soft_face_mask(
        asset["source_points"],
        size=int(asset["crop_size"]),
        feather_px=int(DEFAULT_MASK_FEATHER_PX),
        expand_px=int(DEFAULT_MASK_EXPAND_PX),
    )
    cv2.imwrite(str(mask_path), mask)

    return {
        # profile.json에서 직접 참조하는 핵심 파일 이름들이다.
        "image": source_png_path.name,
        "meta": meta_path.name,
        "mask": mask_path.name,
    }


def iter_requested_views(
    source_dir: Path,
) -> list[tuple[str, str]]:
    # source-dir 안의 고정 파일명 규칙을 "실제로 생성할 view 목록"으로 정규화한다.
    # left/right는 선택 사항이므로, 찾지 못한 항목은 미리 걸러낸다.
    requested_views: list[tuple[str, str | None]] = [
        ("front", find_view_image_path(source_dir, "front")),
        ("left", find_view_image_path(source_dir, "left")),
        ("right", find_view_image_path(source_dir, "right")),
    ]
    resolved_views = [
        (view_name, source_image_path)
        for view_name, source_image_path in requested_views
        if source_image_path
    ]
    if not any(view_name == "front" for view_name, _ in resolved_views):
        raise RuntimeError(f"Missing required front image in source dir: {source_dir}")
    return resolved_views


def resolve_default_view(assets: dict[str, Any], view_names: list[str]) -> str:
    # front가 있으면 그것을 기본 source view로 삼고,
    # 아니면 실제로 존재하는 첫 번째 view를 fallback으로 쓴다.
    return "front" if "front" in assets else view_names[0]


def build_view_select_config() -> dict[str, float | list[float]]:
    # yaw에 따라 어떤 source view를 쓸지에 대한 기준만 별도 helper로 묶는다.
    return {
        "front_yaw_range": [float(DEFAULT_FRONT_YAW_RANGE[0]), float(DEFAULT_FRONT_YAW_RANGE[1])],
        "left_yaw_min": float(DEFAULT_LEFT_YAW_MIN),
        "right_yaw_max": float(DEFAULT_RIGHT_YAW_MAX),
    }


def build_bank_profile(
    *,
    bank_id: str,
    view_names: list[str],
    default_view: str,
    assets: dict[str, Any],
) -> dict[str, Any]:
    # profile.json은 reenact 런타임이 읽는 계약 파일이므로,
    # 실제 실행에 필요한 필드만 모아 따로 조립한다.
    return {
        "avatar_id": bank_id,
        "name": bank_id,
        "views": view_names,
        "default_view": default_view,
        "view_select": build_view_select_config(),
        "assets": assets,
    }


def main() -> None:
    args = parse_args()
    source_dir = resolve_source_dir(args.source_dir)
    # source 폴더 이름이 실제 bank id가 되므로, 이후 reenact 쪽에서는 이 값이 avatar_id처럼 보이게 된다.
    bank_id = resolve_bank_id(source_dir)
    bank_dir = Path(args.output_dir).expanduser().resolve() / bank_id
    bank_dir.mkdir(parents=True, exist_ok=True)

    # bank에는 front만 있을 수도 있고, front + 좌우 측면 view가 함께 있을 수도 있다.
    # left/right 입력은 없어도 허용한다.
    requested_views = iter_requested_views(source_dir)

    # assets는 profile.json의 assets 필드로 거의 그대로 들어간다.
    # 즉 "view 이름 -> 그 view의 파일/요약 메타데이터" 사전이다.
    assets: dict[str, Any] = {}
    view_names: list[str] = []
    for view_name, source_image_path in requested_views:
        assets[view_name] = build_view_entry(
            view_name=view_name,
            source_image_path=source_image_path,
            bank_dir=bank_dir,
        )
        view_names.append(view_name)

    # 최소한 front 하나라도 있어야 bank가 의미가 있다.
    # left/right만 선택적으로 더하는 구조이므로, 비어 있는 경우를 명확히 막는다.
    if not view_names:
        raise RuntimeError("At least one source portrait is required.")

    default_view = resolve_default_view(assets, view_names)
    # profile.json은 reenact 스크립트가 실제 런타임에서 읽는 핵심 계약 파일이다.
    # 폴더 안의 나머지 파일들은 이 profile에서 참조하는 자산들이다.
    profile = build_bank_profile(
        bank_id=bank_id,
        view_names=view_names,
        default_view=default_view,
        assets=assets,
    )

    # reenact 스크립트는 profile.json 하나만 받아도 각 view 자산을 다시 찾을 수 있다.
    # 즉 --avatar-profile path/to/profile.json 만 넘기면 front/left/right meta를 자동으로 로드한다.
    profile_path = bank_dir / PROFILE_FILENAME
    write_json(profile_path, profile)

    print(
        json.dumps(
            {
                "ok": True,
                "bank_id": bank_id,
                "bank_dir": str(bank_dir),
                "profile_path": str(profile_path),
                "views": view_names,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
