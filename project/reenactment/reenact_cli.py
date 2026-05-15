from __future__ import annotations

# keyframe 기반 reenact 실행의 CLI 진입점이다.
# 이 파일은 "사용자가 어떤 옵션으로 실행할지"를 정의하고,
# 실제 처리 흐름은 reenact_pipeline.py로 넘긴다.
#
# 파일 책임을 나누는 현재 구조:
# - reenact_cli.py: CLI 진입점
# - reenact_pipeline.py: 전체 실행 흐름 orchestration
# - reenact_keyframe_cache.py: 계획표 생성, keyframe 선택, cache 계산

import argparse

from .reenact_pipeline import run_keyframe_reenact_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_keyframe_reenact_pipeline(args)


if __name__ == "__main__":
    main()
