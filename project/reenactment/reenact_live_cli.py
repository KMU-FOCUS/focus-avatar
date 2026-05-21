from __future__ import annotations

import argparse

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
    )
    parser.add_argument(
        "--live-source-mode",
        choices=(LIVE_SOURCE_MODE_VIDEO_FILE, LIVE_SOURCE_MODE_STREAM_PAIRS),
        default=LIVE_SOURCE_MODE_VIDEO_FILE,
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

    parser.add_argument("--avatar-bank-dir", nargs="+", required=True)
    parser.add_argument("--avatar-random-seed", type=int, default=0)

    parser.add_argument("--draw-bbox", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--line-thickness", type=int, default=3)

    parser.add_argument("--gpen-model", default=None)
    parser.add_argument("--gpen-provider", default="cpu", choices=("cpu", "coreml", "cuda"))
    parser.add_argument("--gpen-input-size", type=int, default=256)
    parser.add_argument("--key-restorer-mask-expand-px", type=int, default=-1)
    parser.add_argument("--key-restorer-feather-px", type=int, default=8)
    parser.add_argument("--key-restorer-every", type=int, default=1)

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
    _validate_args(args)
    return args


def _require_arg(args: argparse.Namespace, field_name: str) -> None:
    value = getattr(args, field_name, None)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"--{field_name.replace('_', '-')} is required for this mode.")


def _validate_args(args: argparse.Namespace) -> None:
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
    run_live_reenact_pipeline(args)


if __name__ == "__main__":
    main()
