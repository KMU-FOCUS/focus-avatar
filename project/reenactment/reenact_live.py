from __future__ import annotations

# 실시간 reenact 전용 런타임이다.
# 저장된 metadata 전체를 미리 훑는 오프라인 파이프라인과 달리,
# 이 모듈은 "현재 비디오 프레임 1장 + 현재 metadata packet 1개"를 바로 받아
# tracking_id 기준 상태를 유지하면서 합성 결과를 반환한다.

from dataclasses import dataclass, field
import base64
import random
from pathlib import Path
from typing import Any, Iterator, Mapping
import argparse
import json
import sys
import tempfile
import threading

import cv2
import numpy as np

from . import metadata_bbox_utils as overlay_helpers
from .reenact_assets_runtime import discover_avatar_bank_entries, load_avatar_profile_by_id
from .reenact_composite import composite_face
from .reenact_face_planner import bbox_area, filter_faces
from .reenact_keyframe_cache import (
    FrameFacePlan,
    WarpedKeyframe,
    build_single_face_plan,
    build_warped_keyframe,
)
from .reenact_pipeline import load_required_facemap_assets
from .reenact_pipeline import remux_original_audio
from .reenact_restore import load_gpen_keyframe_restorer


LIVE_TARGET_INPUT_MODE_FULL_FRAME = "full_frame"
LIVE_TARGET_INPUT_MODE_METADATA_CROP = "metadata_crop"
LIVE_SOURCE_MODE_VIDEO_FILE = "video_file"
LIVE_SOURCE_MODE_STREAM_PAIRS = "stream_pairs"
LIVE_METADATA_INPUT_MODE_AUTO = "auto"
LIVE_METADATA_INPUT_MODE_BUNDLE_JSON = "bundle_json"
LIVE_METADATA_INPUT_MODE_JSONL = "jsonl"
LIVE_METADATA_INPUT_MODE_STDIN_JSONL = "stdin_jsonl"
LIVE_STREAM_OUTPUT_MODE_STDOUT_JSONL = "stdout_jsonl"
LIVE_STREAM_IMAGE_FORMAT_JPEG = "jpeg"
LIVE_STREAM_IMAGE_FORMAT_PNG = "png"


@dataclass(slots=True)
class LiveReenactConfig:
    # full_frame:
    # - metadata bbox를 그대로 사용한다.
    # metadata_crop:
    # - metadata bbox를 중심 기준으로 조금 더 크게 확장해서 사용한다.
    target_input_mode: str = LIVE_TARGET_INPUT_MODE_FULL_FRAME
    metadata_crop_scale: float = 2.0

    # 최종 합성 bbox 크기만 별도로 조절하고 싶을 때 쓰는 값이다.
    # 1.0이면 metadata bbox를 그대로 쓰고,
    # 1.0보다 크면 얼굴이 더 크게 붙고,
    # 1.0보다 작으면 더 작게 붙는다.
    output_bbox_scale_x: float = 1.0
    output_bbox_scale_y: float = 1.0

    # refresh_every_frames가 1이면 현재 프레임 metadata로 매번 새 warp를 계산한다.
    # 2 이상이면 같은 tracking_id에 대해 그 간격만큼 이전 결과를 재사용한다.
    refresh_every_frames: int = 1

    # tracking_id가 현재 프레임 faces 목록에 없으면 즉시 state를 삭제한다.
    delete_missing_tracks_immediately: bool = True

    # warp coverage mask를 최종 합성 마스크에 반영할지 여부다.
    # False로 두면 reenact_composite.build_face_mask() 쪽 축소 튜닝이 더 직접적으로 보인다.
    use_face_mask_override: bool = False

    # 디버그 시각화 옵션
    draw_bbox: bool = False
    line_thickness: int = 2
    hide_labels: bool = False

    # GPEN 복원 stride 설정
    key_restorer_every: int = 1
    key_restorer_mask_expand_px: int = -1
    key_restorer_feather_px: int = 8

    excluded_tracking_ids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.target_input_mode not in {
            LIVE_TARGET_INPUT_MODE_FULL_FRAME,
            LIVE_TARGET_INPUT_MODE_METADATA_CROP,
        }:
            raise ValueError(
                "target_input_mode must be 'full_frame' or 'metadata_crop'."
            )
        if self.metadata_crop_scale <= 0:
            raise ValueError("metadata_crop_scale must be greater than 0.")
        if self.output_bbox_scale_x <= 0 or self.output_bbox_scale_y <= 0:
            raise ValueError("output_bbox_scale_x/output_bbox_scale_y must be greater than 0.")
        if self.refresh_every_frames < 1:
            raise ValueError("refresh_every_frames must be at least 1.")
        if self.key_restorer_every < 1:
            raise ValueError("key_restorer_every must be at least 1.")


@dataclass(slots=True)
class LiveFaceState:
    face_key: str
    tracking_id: int | None
    avatar_id: str
    source_view: str
    current_warped: WarpedKeyframe | None = None
    last_bbox_xyxy: tuple[float, float, float, float] | None = None
    last_frame_index: int = -1
    last_pts_us: int | None = None
    refresh_count: int = 0


@dataclass(slots=True)
class LiveStreamPacket:
    sequence_id: int
    payload: Mapping[str, Any]


class LatestPacketBuffer:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest_packet: LiveStreamPacket | None = None
        self._closed = False
        self._dropped_since_last_get = 0

    def put(self, packet: LiveStreamPacket) -> None:
        with self._condition:
            if self._latest_packet is not None:
                self._dropped_since_last_get += 1
            self._latest_packet = packet
            self._condition.notify()

    def get(self) -> tuple[LiveStreamPacket | None, int]:
        with self._condition:
            while self._latest_packet is None and not self._closed:
                self._condition.wait()
            if self._latest_packet is None:
                return None, 0
            packet = self._latest_packet
            dropped = self._dropped_since_last_get
            self._latest_packet = None
            self._dropped_since_last_get = 0
            return packet, dropped

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


def _resolve_live_metadata_input_mode(
    metadata_arg: str,
    metadata_input_mode: str,
) -> str:
    normalized = str(metadata_input_mode).strip().lower()
    if normalized == LIVE_METADATA_INPUT_MODE_AUTO:
        if metadata_arg.strip() == "-":
            return LIVE_METADATA_INPUT_MODE_STDIN_JSONL
        suffix = Path(metadata_arg).suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            return LIVE_METADATA_INPUT_MODE_JSONL
        return LIVE_METADATA_INPUT_MODE_BUNDLE_JSON
    if normalized in {
        LIVE_METADATA_INPUT_MODE_BUNDLE_JSON,
        LIVE_METADATA_INPUT_MODE_JSONL,
        LIVE_METADATA_INPUT_MODE_STDIN_JSONL,
    }:
        return normalized
    raise ValueError(f"Unsupported metadata_input_mode: {metadata_input_mode}")


def _iter_jsonl_metadata_packets(
    lines: Iterator[str],
    *,
    source_label: str,
) -> Iterator[Mapping[str, Any]]:
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {source_label} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Expected a JSON object per line in {source_label}, got {type(payload).__name__} "
                f"at line {line_number}."
            )
        yield payload


def _iter_live_metadata_packets(
    metadata_arg: str,
    metadata_input_mode: str,
) -> tuple[Iterator[Mapping[str, Any]], int | None]:
    resolved_mode = _resolve_live_metadata_input_mode(metadata_arg, metadata_input_mode)

    if resolved_mode == LIVE_METADATA_INPUT_MODE_STDIN_JSONL:
        return (
            _iter_jsonl_metadata_packets(iter(sys.stdin.readline, ""), source_label="stdin"),
            None,
        )

    metadata_path = Path(metadata_arg).expanduser().resolve()
    if resolved_mode == LIVE_METADATA_INPUT_MODE_JSONL:
        def iter_from_jsonl_file() -> Iterator[Mapping[str, Any]]:
            with metadata_path.open("r", encoding="utf-8") as handle:
                yield from _iter_jsonl_metadata_packets(handle, source_label=str(metadata_path))

        return iter_from_jsonl_file(), None

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Expected metadata JSON object with a top-level 'frames' array.")
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError("Expected metadata JSON with a top-level 'frames' array.")

    def iter_from_bundle() -> Iterator[Mapping[str, Any]]:
        for frame_index, item in enumerate(raw_frames):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"Expected metadata frame object at index {frame_index}, got {type(item).__name__}."
                )
            yield item

    return iter_from_bundle(), len(raw_frames)


def _build_live_config_from_args(args: argparse.Namespace) -> LiveReenactConfig:
    return LiveReenactConfig(
        target_input_mode=str(args.target_input_mode),
        metadata_crop_scale=float(args.metadata_crop_scale),
        output_bbox_scale_x=float(args.output_bbox_scale_x),
        output_bbox_scale_y=float(args.output_bbox_scale_y),
        refresh_every_frames=int(args.refresh_every_frames),
        delete_missing_tracks_immediately=not bool(args.keep_missing_tracks),
        use_face_mask_override=bool(args.use_face_mask_override),
        draw_bbox=bool(args.draw_bbox),
        line_thickness=int(args.line_thickness),
        hide_labels=bool(args.hide_labels),
        key_restorer_every=int(args.key_restorer_every),
        key_restorer_mask_expand_px=int(args.key_restorer_mask_expand_px),
        key_restorer_feather_px=int(args.key_restorer_feather_px),
    )


def _build_live_renderer_from_args(
    args: argparse.Namespace,
    *,
    config: LiveReenactConfig,
) -> LiveReenactRenderer:
    return LiveReenactRenderer(
        avatar_bank_dir=[str(path) for path in args.avatar_bank_dir],
        avatar_random_seed=int(args.avatar_random_seed),
        config=config,
        gpen_model=args.gpen_model,
        gpen_provider=str(args.gpen_provider),
        gpen_input_size=int(args.gpen_input_size),
    )


def _decode_stream_frame_image(encoded_text: str) -> np.ndarray:
    encoded_bytes = base64.b64decode(encoded_text.encode("ascii"))
    image_buffer = np.frombuffer(encoded_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Failed to decode stream frame image.")
    return frame_bgr


def _extract_stream_frame_bgr(packet: Mapping[str, Any]) -> np.ndarray:
    frame_info = packet.get("frame")
    sources: list[Mapping[str, Any]] = []
    if isinstance(frame_info, Mapping):
        sources.append(frame_info)
    sources.append(packet)

    for source in sources:
        frame_path = source.get("path") if source is frame_info else source.get("frame_path")
        if isinstance(frame_path, str) and frame_path:
            frame_bgr = cv2.imread(str(Path(frame_path).expanduser().resolve()), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise ValueError(f"Failed to read stream frame image from path: {frame_path}")
            return frame_bgr

        for key in (
            "jpeg_base64",
            "png_base64",
            "image_base64",
        ) if source is frame_info else (
            "frame_jpeg_base64",
            "frame_png_base64",
            "frame_image_base64",
            "image_base64",
        ):
            value = source.get(key)
            if isinstance(value, str) and value:
                return _decode_stream_frame_image(value)

    raise ValueError(
        "Stream packet is missing frame image data. Pass frame_path, frame_jpeg_base64, frame_png_base64, "
        "or frame.image_base64."
    )


def _extract_stream_metadata_packet(packet: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("metadata", "metadata_packet", "frame_metadata"):
        value = packet.get(key)
        if isinstance(value, Mapping):
            metadata_packet = dict(value)
            if "pts_us" not in metadata_packet and packet.get("pts_us") is not None:
                metadata_packet["pts_us"] = packet.get("pts_us")
            return metadata_packet
    if isinstance(packet.get("faces"), list) or isinstance(packet.get("frames"), list):
        return packet
    raise ValueError("Stream packet is missing metadata object.")


def _encode_output_frame_bgr(
    frame_bgr: np.ndarray,
    *,
    image_format: str,
    jpeg_quality: int,
) -> tuple[str, str]:
    normalized_format = str(image_format).strip().lower()
    if normalized_format == LIVE_STREAM_IMAGE_FORMAT_PNG:
        ok, encoded = cv2.imencode(".png", frame_bgr)
        if not ok:
            raise RuntimeError("Failed to encode output frame as PNG.")
        return base64.b64encode(encoded.tobytes()).decode("ascii"), LIVE_STREAM_IMAGE_FORMAT_PNG

    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(jpeg_quality)))],
    )
    if not ok:
        raise RuntimeError("Failed to encode output frame as JPEG.")
    return base64.b64encode(encoded.tobytes()).decode("ascii"), LIVE_STREAM_IMAGE_FORMAT_JPEG


def _emit_stream_result_packet(
    *,
    input_packet: LiveStreamPacket,
    output_frame_bgr: np.ndarray,
    dropped_inputs_while_busy: int,
    output_image_format: str,
    output_jpeg_quality: int,
) -> None:
    metadata_packet = _extract_stream_metadata_packet(input_packet.payload)
    encoded_frame, resolved_image_format = _encode_output_frame_bgr(
        output_frame_bgr,
        image_format=output_image_format,
        jpeg_quality=output_jpeg_quality,
    )
    result_packet = {
        "type": "frame_result",
        "sequence_id": int(input_packet.sequence_id),
        "source_packet_id": input_packet.payload.get("packet_id"),
        "pts_us": metadata_packet.get("pts_us"),
        "dropped_inputs_while_busy": int(dropped_inputs_while_busy),
        "image_format": resolved_image_format,
        "frame_image_base64": encoded_frame,
    }
    sys.stdout.write(json.dumps(result_packet, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run_live_reenact_stream_pipeline(args: argparse.Namespace) -> None:
    if str(args.stream_output_mode) != LIVE_STREAM_OUTPUT_MODE_STDOUT_JSONL:
        raise ValueError("Only stdout_jsonl stream output mode is currently supported.")

    config = _build_live_config_from_args(args)
    renderer = _build_live_renderer_from_args(args, config=config)
    packet_buffer = LatestPacketBuffer()
    worker_error: list[BaseException] = []
    processed_count_holder = {"processed": 0}
    dropped_input_holder = {"dropped": 0}

    def worker_loop() -> None:
        try:
            while True:
                stream_packet, dropped_inputs = packet_buffer.get()
                if stream_packet is None:
                    break
                frame_bgr = _extract_stream_frame_bgr(stream_packet.payload)
                metadata_packet = _extract_stream_metadata_packet(stream_packet.payload)
                output_frame = renderer.process_frame(frame_bgr, metadata_packet)
                _emit_stream_result_packet(
                    input_packet=stream_packet,
                    output_frame_bgr=output_frame,
                    dropped_inputs_while_busy=dropped_inputs,
                    output_image_format=str(args.stream_output_image_format),
                    output_jpeg_quality=int(args.stream_output_jpeg_quality),
                )
                processed_count_holder["processed"] += 1
                dropped_input_holder["dropped"] += int(dropped_inputs)
        except BaseException as exc:  # pragma: no cover - defensive for stream shutdown paths
            worker_error.append(exc)
            packet_buffer.close()

    worker_thread = threading.Thread(target=worker_loop, name="live-reenact-stream-worker", daemon=True)
    worker_thread.start()

    try:
        for sequence_id, payload in enumerate(
            _iter_jsonl_metadata_packets(iter(sys.stdin.readline, ""), source_label="stdin"),
            start=1,
        ):
            packet_buffer.put(
                LiveStreamPacket(
                    sequence_id=sequence_id,
                    payload=payload,
                )
            )
    finally:
        packet_buffer.close()
        worker_thread.join()

    if worker_error:
        raise worker_error[0]

    print(
        json.dumps(
            {
                "type": "stream_summary",
                "processed_frames": int(processed_count_holder["processed"]),
                "dropped_inputs_while_busy": int(dropped_input_holder["dropped"]),
                "gpen_load_error": renderer.gpen_load_error,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def _normalize_live_metadata_frame(metadata_packet: Mapping[str, Any] | None) -> Mapping[str, Any]:
    # 실시간 입력은 보통 frame dict 하나가 오지만,
    # {"frames": [frame]} 형태로 감싸서 오는 경우도 허용한다.
    if not isinstance(metadata_packet, Mapping):
        return {}
    if isinstance(metadata_packet.get("faces"), list):
        return metadata_packet
    frames = metadata_packet.get("frames")
    if isinstance(frames, list):
        for item in frames:
            if isinstance(item, Mapping):
                return item
    return {}


def _tracking_id_from_face_key(face_key: str) -> int | None:
    if not face_key.startswith("track:"):
        return None
    try:
        return int(face_key.split(":", 1)[1])
    except (TypeError, ValueError):
        return None


class LiveReenactRenderer:
    def __init__(
        self,
        *,
        avatar_bank_dir: list[str] | tuple[str, ...],
        avatar_random_seed: int = 0,
        config: LiveReenactConfig | None = None,
        gpen_model: str | Path | None = None,
        gpen_provider: str = "cpu",
        gpen_input_size: int = 256,
    ) -> None:
        self.config = config if config is not None else LiveReenactConfig()
        self.frame_index = 0
        self.face_states: dict[str, LiveFaceState] = {}
        self.smooth_state: dict[int, np.ndarray] = {}
        self.untracked_face_centers: dict[str, np.ndarray] = {}
        self.avatar_assignment_by_face: dict[str, str] = {}
        self.avatar_rng = random.Random(int(avatar_random_seed))
        self.avatar_view_cache: dict[str, dict[str, Any]] = {}

        self.mean_face, self.shape_basis, self.blendshape_basis = load_required_facemap_assets()

        avatar_inputs = [str(Path(path).expanduser().resolve()) for path in avatar_bank_dir]
        self.avatar_profile_paths_by_id = discover_avatar_bank_entries(avatar_inputs)
        self.avatar_ids = sorted(self.avatar_profile_paths_by_id.keys())
        if not self.avatar_ids:
            raise RuntimeError("No avatar folders with profile.json were found under avatar_bank_dir inputs.")

        self.avatar_profile_cache: dict[str, dict[str, Any]] = {}

        self.gpen_keyframe_restorer = None
        self.gpen_load_error: str | None = None
        if gpen_model:
            try:
                self.gpen_keyframe_restorer = load_gpen_keyframe_restorer(
                    str(Path(gpen_model).expanduser().resolve()),
                    provider=str(gpen_provider),
                    input_size=int(gpen_input_size),
                )
            except Exception as exc:
                self.gpen_load_error = str(exc)

    def load_avatar_profile_for_id(self, avatar_id: str) -> dict[str, Any]:
        cached_profile = self.avatar_profile_cache.get(avatar_id)
        if cached_profile is not None:
            return cached_profile
        profile = load_avatar_profile_by_id(self.avatar_profile_paths_by_id, avatar_id)
        self.avatar_profile_cache[avatar_id] = profile
        return profile

    def reset(self, *, preserve_avatar_assignments: bool = True) -> None:
        self.frame_index = 0
        self.face_states.clear()
        self.smooth_state.clear()
        self.untracked_face_centers.clear()
        self.avatar_view_cache.clear()
        if not preserve_avatar_assignments:
            self.avatar_assignment_by_face.clear()

    def _adapt_plan_bbox(
        self,
        plan: FrameFacePlan,
        *,
        frame_w: int,
        frame_h: int,
    ) -> FrameFacePlan | None:
        box = tuple(int(round(v)) for v in plan.bbox_xyxy)
        if self.config.target_input_mode == LIVE_TARGET_INPUT_MODE_METADATA_CROP:
            box = overlay_helpers.scale_box(
                box,
                scale_x=float(self.config.metadata_crop_scale),
                scale_y=float(self.config.metadata_crop_scale),
            )
        box = overlay_helpers.scale_box(
            box,
            scale_x=float(self.config.output_bbox_scale_x),
            scale_y=float(self.config.output_bbox_scale_y),
        )
        clamped = overlay_helpers.clamp_box(box, frame_w, frame_h)
        if clamped is None:
            return None
        return FrameFacePlan(
            frame_index=plan.frame_index,
            face_key=plan.face_key,
            tracking_id=plan.tracking_id,
            bbox_xyxy=tuple(float(v) for v in clamped),
            coeff_264=plan.coeff_264,
            avatar_id=plan.avatar_id,
            source_view=plan.source_view,
        )

    def _should_refresh_warp(self, state: LiveFaceState | None, plan: FrameFacePlan) -> bool:
        if state is None or state.current_warped is None:
            return True
        if state.avatar_id != plan.avatar_id or state.source_view != plan.source_view:
            return True
        frame_gap = self.frame_index - state.last_frame_index
        return frame_gap >= int(self.config.refresh_every_frames)

    def _build_or_reuse_warp(self, state: LiveFaceState | None, plan: FrameFacePlan) -> WarpedKeyframe:
        if state is not None and state.current_warped is not None and not self._should_refresh_warp(state, plan):
            return state.current_warped

        refresh_count = 0 if state is None else int(state.refresh_count)
        restorer_stride = max(1, int(self.config.key_restorer_every))
        should_restore_keyframe = (refresh_count % restorer_stride) == 0
        return build_warped_keyframe(
            plan=plan,
            mean_face=self.mean_face,
            shape_basis=self.shape_basis,
            blendshape_basis=self.blendshape_basis,
            load_avatar_profile_for_id=self.load_avatar_profile_for_id,
            avatar_view_cache=self.avatar_view_cache,
            gpen_keyframe_restorer=self.gpen_keyframe_restorer,
            should_restore_keyframe=should_restore_keyframe,
            key_restorer_mask_expand_px=int(self.config.key_restorer_mask_expand_px),
            key_restorer_feather_px=int(self.config.key_restorer_feather_px),
        )

    def _cleanup_missing_face_states(self, seen_face_keys: set[str]) -> None:
        if not self.config.delete_missing_tracks_immediately:
            return

        stale_keys = [face_key for face_key in self.face_states.keys() if face_key not in seen_face_keys]
        for face_key in stale_keys:
            tracking_id = _tracking_id_from_face_key(face_key)
            if tracking_id is not None:
                self.smooth_state.pop(tracking_id, None)
            self.untracked_face_centers.pop(face_key, None)
            self.face_states.pop(face_key, None)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        metadata_packet: Mapping[str, Any] | None,
    ) -> np.ndarray:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
            raise ValueError("frame_bgr must be a HxWxC uint8 image.")

        metadata_frame = _normalize_live_metadata_frame(metadata_packet)
        raw_faces = metadata_frame.get("faces", []) or []
        faces = filter_faces(
            raw_faces if isinstance(raw_faces, list) else [],
            excluded_tracking_ids=set(int(v) for v in self.config.excluded_tracking_ids),
        )
        selected_faces = sorted(faces, key=bbox_area, reverse=True)

        output_frame = frame_bgr.copy()
        seen_face_keys: set[str] = set()
        used_untracked_face_keys: set[str] = set()
        frame_h, frame_w = output_frame.shape[:2]
        frame_pts_us = metadata_frame.get("pts_us")
        pts_us_int = int(frame_pts_us) if frame_pts_us is not None else None

        for slot_index, selected in enumerate(selected_faces):
            plan = build_single_face_plan(
                frame_index=self.frame_index,
                selected=selected,
                slot_index=slot_index,
                smooth_state=self.smooth_state,
                untracked_face_centers=self.untracked_face_centers,
                used_untracked_face_keys=used_untracked_face_keys,
                avatar_ids=self.avatar_ids,
                load_avatar_profile_for_id=self.load_avatar_profile_for_id,
                avatar_assignment_by_face=self.avatar_assignment_by_face,
                avatar_rng=self.avatar_rng,
            )
            if plan is None:
                continue

            plan = self._adapt_plan_bbox(plan, frame_w=frame_w, frame_h=frame_h)
            if plan is None:
                continue

            seen_face_keys.add(plan.face_key)
            state = self.face_states.get(plan.face_key)
            try:
                warped = self._build_or_reuse_warp(state, plan)
            except Exception:
                if state is None or state.current_warped is None:
                    continue
                warped = state.current_warped

            output_frame = composite_face(
                output_frame,
                warped.face_bgr,
                plan.bbox_xyxy,
                warped.crop_points,
                face_mask_override=warped.mask_uint8 if self.config.use_face_mask_override else None,
            )

            if self.config.draw_bbox:
                overlay_helpers.draw_bbox_overlay(
                    output_frame,
                    tuple(int(round(v)) for v in plan.bbox_xyxy),
                    tracking_id=plan.tracking_id,
                    line_thickness=int(self.config.line_thickness),
                    hide_labels=bool(self.config.hide_labels),
                )

            refresh_count = 1 if state is None else int(state.refresh_count)
            if state is not None and state.current_warped is warped and not self._should_refresh_warp(state, plan):
                refresh_count = int(state.refresh_count)
            elif state is not None:
                refresh_count = int(state.refresh_count) + 1

            self.face_states[plan.face_key] = LiveFaceState(
                face_key=plan.face_key,
                tracking_id=plan.tracking_id,
                avatar_id=plan.avatar_id,
                source_view=plan.source_view,
                current_warped=warped,
                last_bbox_xyxy=plan.bbox_xyxy,
                last_frame_index=self.frame_index,
                last_pts_us=pts_us_int,
                refresh_count=refresh_count,
            )

        self._cleanup_missing_face_states(seen_face_keys)
        self.frame_index += 1
        return output_frame


def run_live_reenact_video_pipeline(args: argparse.Namespace) -> None:
    metadata_packets, metadata_frames_total = _iter_live_metadata_packets(
        str(args.metadata),
        str(args.metadata_input_mode),
    )
    metadata_packet_iter = iter(metadata_packets)

    config = _build_live_config_from_args(args)
    renderer = _build_live_renderer_from_args(args, config=config)

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

    input_fps = 30.0
    width = 0
    height = 0
    frames_rendered = 0
    metadata_frames_used = 0
    audio_remuxed = False
    audio_remux_error: str | None = None

    try:
        cap = cv2.VideoCapture(str(Path(args.video).expanduser().resolve()))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {args.video}")

        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(input_fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create output video: {output_video_path}")

        frame_index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            try:
                metadata_packet = next(metadata_packet_iter)
            except StopIteration:
                break

            output_frame = renderer.process_frame(frame_bgr, metadata_packet)
            writer.write(output_frame)
            frames_rendered += 1
            metadata_frames_used += 1
            frame_index += 1
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()

    if frames_rendered > 0:
        remuxed, remux_error = remux_original_audio(
            video_only_path=output_video_path,
            source_video_path=args.video,
            output_video_path=final_output_video_path,
        )
        if remuxed:
            audio_remuxed = True
        else:
            audio_remux_error = remux_error
            output_video_path.replace(final_output_video_path)
    else:
        output_video_path.replace(final_output_video_path)

    print(
        json.dumps(
            {
                "frames_rendered": frames_rendered,
                "metadata_frames_used": metadata_frames_used,
                "metadata_frames_total": metadata_frames_total,
                "audio_remuxed": audio_remuxed,
                "audio_remux_error": audio_remux_error,
                "gpen_load_error": renderer.gpen_load_error,
                "output_video": str(final_output_video_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_live_reenact_pipeline(args: argparse.Namespace) -> None:
    if str(args.live_source_mode) == LIVE_SOURCE_MODE_STREAM_PAIRS:
        run_live_reenact_stream_pipeline(args)
        return
    run_live_reenact_video_pipeline(args)


__all__ = [
    "LIVE_TARGET_INPUT_MODE_FULL_FRAME",
    "LIVE_TARGET_INPUT_MODE_METADATA_CROP",
    "LIVE_SOURCE_MODE_VIDEO_FILE",
    "LIVE_SOURCE_MODE_STREAM_PAIRS",
    "LIVE_METADATA_INPUT_MODE_AUTO",
    "LIVE_METADATA_INPUT_MODE_BUNDLE_JSON",
    "LIVE_METADATA_INPUT_MODE_JSONL",
    "LIVE_METADATA_INPUT_MODE_STDIN_JSONL",
    "LIVE_STREAM_OUTPUT_MODE_STDOUT_JSONL",
    "LIVE_STREAM_IMAGE_FORMAT_JPEG",
    "LIVE_STREAM_IMAGE_FORMAT_PNG",
    "LiveFaceState",
    "LiveReenactConfig",
    "LiveReenactRenderer",
    "run_live_reenact_pipeline",
    "run_live_reenact_stream_pipeline",
    "run_live_reenact_video_pipeline",
]
