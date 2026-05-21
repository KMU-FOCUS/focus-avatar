from __future__ import annotations

# reenact 실행의 통합 CLI 진입점이다.
# 공통 인자는 여기서 받고,
# - file 모드면 기존 keyframe reenact 파이프라인으로
# - live 모드면 frame-by-frame live reenact 파이프라인으로 넘긴다.

import argparse
import sys

from .reenact_live import (
    LIVE_TARGET_INPUT_MODE_FULL_FRAME,
    LIVE_TARGET_INPUT_MODE_METADATA_CROP,
    run_live_reenact_video_pipeline,
)
from .reenact_pipeline import run_keyframe_reenact_pipeline

RUN_MODE_FILE = "file"
RUN_MODE_LIVE = "live"
RUN_MODE_PROMPT = "prompt"


def _resolve_run_mode(run_mode: str) -> str:
    normalized = str(run_mode).strip().lower()
    if normalized in {RUN_MODE_FILE, RUN_MODE_LIVE}:
        return normalized
    if normalized != RUN_MODE_PROMPT:
        raise ValueError(f"Unsupported run mode: {run_mode}")

    if not sys.stdin.isatty():
        return RUN_MODE_FILE

    print("Choose reenact mode:")
    print(f"  1) {RUN_MODE_FILE}  - metadata/video 전체를 받아 keyframe 기반으로 처리")
    print(f"  2) {RUN_MODE_LIVE}  - metadata를 프레임마다 바로 반영하는 live 처리")
    while True:
        choice = input("Enter 1 or 2 [default: 1]: ").strip().lower()
        if choice in {"", "1", RUN_MODE_FILE, "f"}:
            return RUN_MODE_FILE
        if choice in {"2", RUN_MODE_LIVE, "l"}:
            return RUN_MODE_LIVE
        print("Please enter 1 or 2.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-mode",
        choices=(RUN_MODE_PROMPT, RUN_MODE_FILE, RUN_MODE_LIVE),
        default=RUN_MODE_PROMPT,
        help="Choose file pipeline or frame-by-frame live pipeline. Default prompts in TTY and falls back to file.",
    )

    # 입출력 파일
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-video", required=True)

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
    args.run_mode = _resolve_run_mode(args.run_mode)
    return args


def main() -> None:
    args = parse_args()
    if args.run_mode == RUN_MODE_LIVE:
        run_live_reenact_video_pipeline(args)
        return
    run_keyframe_reenact_pipeline(args)


if __name__ == "__main__":
    main()
