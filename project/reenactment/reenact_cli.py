from __future__ import annotations

# reenact 실행의 통합 CLI 진입점이다.
# 공통 인자는 여기서 받고,
# - file 모드면 기존 keyframe reenact 파이프라인으로
# - live 모드면 frame-by-frame live reenact 파이프라인으로 넘긴다.

import argparse
import sys

from .reenact_live import (
    LIVE_METADATA_INPUT_MODE_AUTO,
    LIVE_METADATA_INPUT_MODE_BUNDLE_JSON,
    LIVE_METADATA_INPUT_MODE_JSONL,
    LIVE_METADATA_INPUT_MODE_STDIN_JSONL,
    LIVE_SOURCE_MODE_STREAM_PAIRS,
    LIVE_SOURCE_MODE_VIDEO_FILE,
    LIVE_STREAM_IMAGE_FORMAT_JPEG,
    LIVE_STREAM_IMAGE_FORMAT_PNG,
    LIVE_STREAM_OUTPUT_MODE_STDOUT_JSONL,
    LIVE_TARGET_INPUT_MODE_FULL_FRAME,
    LIVE_TARGET_INPUT_MODE_METADATA_CROP,
    run_live_reenact_pipeline,
)
from .reenact_pipeline import run_keyframe_reenact_pipeline

RUN_MODE_FILE = "file"
RUN_MODE_LIVE = "live"
RUN_MODE_PROMPT = "prompt"


def _resolve_run_mode(
    run_mode: str,
    *,
    metadata: str | None = None,
    metadata_input_mode: str = LIVE_METADATA_INPUT_MODE_AUTO,
    live_source_mode: str = LIVE_SOURCE_MODE_VIDEO_FILE,
) -> str:
    normalized = str(run_mode).strip().lower()
    if normalized in {RUN_MODE_FILE, RUN_MODE_LIVE}:
        return normalized
    if normalized != RUN_MODE_PROMPT:
        raise ValueError(f"Unsupported run mode: {run_mode}")

    metadata_path = (metadata or "").strip()
    if live_source_mode == LIVE_SOURCE_MODE_STREAM_PAIRS:
        return RUN_MODE_LIVE
    if metadata_input_mode in {LIVE_METADATA_INPUT_MODE_JSONL, LIVE_METADATA_INPUT_MODE_STDIN_JSONL}:
        return RUN_MODE_LIVE
    if metadata_input_mode == LIVE_METADATA_INPUT_MODE_AUTO and metadata_path == "-":
        return RUN_MODE_LIVE

    if not sys.stdin.isatty():
        return RUN_MODE_LIVE

    print("Choose reenact mode:")
    print(f"  1) {RUN_MODE_LIVE}  - metadata를 프레임마다 바로 반영하는 live 처리")
    print(f"  2) {RUN_MODE_FILE}  - metadata/video 전체를 받아 keyframe 기반으로 처리")
    while True:
        choice = input("Enter 1 or 2 [default: 1]: ").strip().lower()
        if choice in {"", "1", RUN_MODE_LIVE, "l"}:
            return RUN_MODE_LIVE
        if choice in {"2", RUN_MODE_FILE, "f"}:
            return RUN_MODE_FILE
        print("Please enter 1 or 2.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-mode",
        choices=(RUN_MODE_PROMPT, RUN_MODE_FILE, RUN_MODE_LIVE),
        default=RUN_MODE_LIVE,
        help="Choose file pipeline or frame-by-frame live pipeline. Default is live; use prompt or file when needed.",
    )

    # 입출력 파일
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--output-video", default=None)
    parser.add_argument(
        "--metadata-input-mode",
        choices=(
            LIVE_METADATA_INPUT_MODE_AUTO,
            LIVE_METADATA_INPUT_MODE_BUNDLE_JSON,
            LIVE_METADATA_INPUT_MODE_JSONL,
            LIVE_METADATA_INPUT_MODE_STDIN_JSONL,
        ),
        default=LIVE_METADATA_INPUT_MODE_AUTO,
        help="For live mode, read metadata as full bundle JSON, JSONL packets, or stdin JSONL stream.",
    )
    parser.add_argument(
        "--live-source-mode",
        choices=(LIVE_SOURCE_MODE_VIDEO_FILE, LIVE_SOURCE_MODE_STREAM_PAIRS),
        default=LIVE_SOURCE_MODE_VIDEO_FILE,
        help="For live mode, use an internal video file reader or consume frame+metadata pairs from stdin.",
    )
    parser.add_argument(
        "--stream-output-mode",
        choices=(LIVE_STREAM_OUTPUT_MODE_STDOUT_JSONL,),
        default=LIVE_STREAM_OUTPUT_MODE_STDOUT_JSONL,
    )
    parser.add_argument(
        "--stream-output-image-format",
        choices=(LIVE_STREAM_IMAGE_FORMAT_JPEG, LIVE_STREAM_IMAGE_FORMAT_PNG),
        default=LIVE_STREAM_IMAGE_FORMAT_JPEG,
    )
    parser.add_argument("--stream-output-jpeg-quality", type=int, default=90)

    # avatar 입력 방식
    # - avatar-bank-dir: avatar_bank 경로
    #   메타데이터에서 새 얼굴이 처음 보일 때 avatar_id를 배정
    parser.add_argument("--avatar-bank-dir", nargs="+", required=True)
    # - avatar-random-seed: 여러 avatar bank를 쓸 때 배정 순서를 고정
    parser.add_argument("--avatar-random-seed", type=int, default=0)

    # 디버그 시각화 옵션
    # - 실제 합성 품질에는 영향을 주지 않고,
    #   bbox/landmark/mask가 어디에 그려지는지 눈으로 확인할 때만 사용한다.
    parser.add_argument("--draw-bbox", action="store_true")
    parser.add_argument("--draw-landmarks", action="store_true")
    parser.add_argument("--landmark-radius", type=int, default=2)
    parser.add_argument("--draw-mask", action="store_true")
    parser.add_argument("--mask-alpha", type=float, default=0.35)
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--line-thickness", type=int, default=3)

    # 얼굴복원모델(선택사항)
    # - gpen-* : GPEN keyframe 복원 설정
    parser.add_argument("--gpen-model", default=None)
    parser.add_argument("--gpen-provider", default="cpu", choices=("cpu", "coreml", "cuda"))
    parser.add_argument("--gpen-input-size", type=int, default=256)
    parser.add_argument("--key-restorer-mask-expand-px", type=int, default=-1)
    parser.add_argument("--key-restorer-feather-px", type=int, default=8)
    parser.add_argument("--key-restorer-every", type=int, default=1)

    # live 모드 전용 세부 옵션
    parser.add_argument(
        "--target-input-mode",
        choices=(LIVE_TARGET_INPUT_MODE_FULL_FRAME, LIVE_TARGET_INPUT_MODE_METADATA_CROP),
        default=LIVE_TARGET_INPUT_MODE_FULL_FRAME,
    )
    parser.add_argument("--metadata-crop-scale", type=float, default=2.0)
    parser.add_argument("--output-bbox-scale-x", type=float, default=1.0)
    parser.add_argument("--output-bbox-scale-y", type=float, default=1.0)
    parser.add_argument("--refresh-every-frames", type=int, default=1)
    parser.add_argument("--keep-missing-tracks", action="store_true")
    parser.add_argument("--use-face-mask-override", action="store_true")

    args = parser.parse_args()
    args.run_mode = _resolve_run_mode(
        args.run_mode,
        metadata=args.metadata,
        metadata_input_mode=str(args.metadata_input_mode),
        live_source_mode=str(args.live_source_mode),
    )
    _validate_args(args)
    return args


def _require_arg(args: argparse.Namespace, field_name: str) -> None:
    value = getattr(args, field_name, None)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"--{field_name.replace('_', '-')} is required for this mode.")


def _validate_args(args: argparse.Namespace) -> None:
    if args.run_mode == RUN_MODE_FILE:
        for field_name in ("metadata", "video", "output_video"):
            _require_arg(args, field_name)
        return

    if str(args.live_source_mode) == LIVE_SOURCE_MODE_VIDEO_FILE:
        for field_name in ("metadata", "video", "output_video"):
            _require_arg(args, field_name)
        return

    if args.metadata is not None and str(args.metadata).strip():
        raise ValueError("Do not pass --metadata when --live-source-mode stream_pairs is used.")
    if args.video is not None and str(args.video).strip():
        raise ValueError("Do not pass --video when --live-source-mode stream_pairs is used.")
    if args.output_video is not None and str(args.output_video).strip():
        raise ValueError("Do not pass --output-video when --live-source-mode stream_pairs is used.")


def main() -> None:
    args = parse_args()
    if args.run_mode == RUN_MODE_LIVE:
        run_live_reenact_pipeline(args)
        return
    run_keyframe_reenact_pipeline(args)


if __name__ == "__main__":
    main()
