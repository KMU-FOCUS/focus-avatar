from __future__ import annotations

# reenact 런타임에서 직접 쓰는 source/avatar 자산 로더 모음이다.
# avatar bank 탐색, profile/view 선택, precomputed meta 로딩처럼
# 프레임 렌더링 중 자주 필요한 가벼운 IO 헬퍼만 여기에 둔다.

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

CROP_SIZE = 256
AVATAR_PROFILE_FILENAME = "profile.json"


def load_json(path: str | Path) -> Any:
    # 런타임에서 반복해서 쓰는 가장 작은 JSON 로더다.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    # npz 핸들을 즉시 닫고, 이후에는 메모리 안 dict만 다루게 한다.
    with np.load(Path(path), allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _extract_source_crop_and_points(
    payload: Mapping[str, np.ndarray],
    *,
    asset_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    # reenact runtime이 실제로 기대하는 최소 배열 두 개만 검증해서 꺼낸다.
    source_crop = payload.get("source_crop_bgr")
    source_points = payload.get("source_points")
    if source_crop is None or source_points is None:
        raise RuntimeError(
            f"{asset_label} is missing required 'source_crop_bgr' or 'source_points' arrays."
        )
    return (
        np.asarray(source_crop, dtype=np.uint8),
        np.asarray(source_points, dtype=np.float32),
    )


def load_precomputed_source_asset(path: str | Path) -> dict[str, np.ndarray]:
    # 단일 source-meta npz를 runtime 입력 형태로 정규화한다.
    payload = _load_npz(path)
    source_crop, source_points = _extract_source_crop_and_points(
        payload,
        asset_label=str(Path(path)),
    )
    return {
        "source_crop_bgr": source_crop,
        "source_points": source_points,
    }


def coeff_to_pose_radians(coeff_264: Sequence[float] | np.ndarray) -> dict[str, float]:
    # FaceMap coeff 꼬리에서 pose 3축만 꺼내어 radians로 돌려준다.
    coeff = np.asarray(coeff_264, dtype=np.float32).reshape(-1)
    if coeff.size < 261:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    return {
        "pitch": float(coeff[258]) * float(np.pi / 2.0),
        "yaw": float(coeff[259]) * float(np.pi / 2.0),
        "roll": float(coeff[260]) * float(np.pi / 2.0),
    }


def load_avatar_profile(profile_path: str | Path) -> dict[str, Any]:
    # profile 원본 내용에 runtime용 내부 메타데이터만 얇게 덧붙인다.
    profile_file = Path(profile_path).expanduser().resolve()
    profile = load_json(profile_file)
    if not isinstance(profile, dict):
        raise RuntimeError(f"Invalid avatar profile JSON: {profile_file}")
    profile["_profile_dir"] = str(profile_file.parent)
    return profile


def discover_avatar_bank_entries(bank_inputs: Sequence[str | Path]) -> dict[str, str]:
    # avatar bank 루트/개별 avatar 폴더를 받아 avatar_id -> profile 경로 맵을 만든다.
    # 여기서는 profile 내용을 읽지 않고, 후보 경로만 가볍게 수집한다.
    profile_paths_by_id: dict[str, str] = {}

    for raw_path in bank_inputs:
        candidate = Path(raw_path).expanduser().resolve()
        avatar_dirs: list[Path] = []

        direct_profile = candidate / AVATAR_PROFILE_FILENAME
        if direct_profile.exists():
            avatar_dirs.append(candidate)
        elif candidate.is_dir():
            avatar_dirs.extend(
                child
                for child in sorted(candidate.iterdir())
                if child.is_dir() and (child / AVATAR_PROFILE_FILENAME).exists()
            )

        for avatar_dir in avatar_dirs:
            base_avatar_id = avatar_dir.name
            avatar_id = base_avatar_id
            suffix = 2
            while avatar_id in profile_paths_by_id:
                avatar_id = f"{base_avatar_id}_{suffix}"
                suffix += 1
            profile_paths_by_id[avatar_id] = str((avatar_dir / AVATAR_PROFILE_FILENAME).resolve())

    return profile_paths_by_id


def load_avatar_profile_by_id(
    avatar_profile_paths_by_id: Mapping[str, str],
    avatar_id: str,
) -> dict[str, Any]:
    # 배정된 avatar_id 하나를 실제 profile payload로 지연 로드한다.
    profile_path = avatar_profile_paths_by_id.get(avatar_id)
    if not isinstance(profile_path, str):
        raise RuntimeError(f"Unknown avatar_id '{avatar_id}'.")
    profile = load_avatar_profile(profile_path)
    profile["avatar_id"] = avatar_id
    return profile


def select_avatar_view(profile: Mapping[str, Any], yaw_radians: float) -> str:
    # driving frame의 yaw를 기준으로 front/left/right 중 어떤 source view를 쓸지 고른다.
    assets = profile.get("assets")
    if not isinstance(assets, Mapping) or not assets:
        raise RuntimeError("Avatar profile is missing an 'assets' object.")

    default_view = str(profile.get("default_view") or next(iter(assets.keys())))
    available_views = set(str(name) for name in assets.keys())
    view_select = profile.get("view_select") if isinstance(profile.get("view_select"), Mapping) else {}

    yaw_degrees = float(np.degrees(yaw_radians))
    front_range = view_select.get("front_yaw_range")
    if (
        "front" in available_views
        and isinstance(front_range, list)
        and len(front_range) >= 2
        and float(front_range[0]) <= yaw_degrees <= float(front_range[1])
    ):
        return "front"

    left_min = float(view_select.get("left_yaw_min", 32.0))
    right_max = float(view_select.get("right_yaw_max", -32.0))
    if yaw_degrees >= left_min and "left" in available_views:
        return "left"
    if yaw_degrees <= right_max and "right" in available_views:
        return "right"
    return default_view


def _profile_dir_from_profile(profile: Mapping[str, Any]) -> Path:
    # profile 내부에 기록해 둔 절대 디렉터리를 꺼내서 후속 asset path 계산에 쓴다.
    profile_dir = profile.get("_profile_dir")
    if not isinstance(profile_dir, str) or not profile_dir:
        raise RuntimeError("Avatar profile is missing internal '_profile_dir' metadata.")
    return Path(profile_dir)


def load_avatar_view_assets(profile: Mapping[str, Any], view_name: str) -> dict[str, Any]:
    # 선택된 view의 precomputed meta/mask를 읽어 실제 warp 입력 배열로 바꾼다.
    profile_dir = _profile_dir_from_profile(profile)
    assets = profile.get("assets")
    if not isinstance(assets, Mapping):
        raise RuntimeError("Avatar profile is missing an 'assets' mapping.")
    view = assets.get(view_name)
    if not isinstance(view, Mapping):
        raise RuntimeError(f"Avatar profile does not define view '{view_name}'.")

    meta_name = view.get("meta")
    mask_name = view.get("mask")
    if not isinstance(meta_name, str):
        raise RuntimeError(f"Avatar view '{view_name}' is missing a meta entry.")

    meta_path = profile_dir / meta_name
    if not meta_path.exists():
        raise RuntimeError(f"Avatar view '{view_name}' meta file was not found: {meta_path}")
    source_payload = _load_npz(meta_path)
    source_crop, source_points = _extract_source_crop_and_points(
        source_payload,
        asset_label=f"{view_name} meta ({meta_path})",
    )

    result: dict[str, Any] = {
        "view_name": view_name,
        "source_crop_bgr": source_crop,
        "source_points": source_points,
    }
    if isinstance(mask_name, str):
        mask_path = profile_dir / mask_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            result["source_mask"] = mask
    return result


__all__ = [
    "CROP_SIZE",
    "coeff_to_pose_radians",
    "discover_avatar_bank_entries",
    "load_avatar_profile",
    "load_avatar_profile_by_id",
    "load_avatar_view_assets",
    "load_json",
    "load_precomputed_source_asset",
    "select_avatar_view",
]
