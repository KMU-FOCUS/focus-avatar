# focus-avatar

`focus_avatar`는 source avatar 이미지를 미리 전처리한 뒤, driving metadata의 3DMM coefficient를 사용해 target 얼굴 형상을 복원하고, source avatar를 landmark 기반 warp로 변형하여 원본 영상에 합성하는 Python 파이프라인입니다.

이 프로젝트는 생성형 모델로 얼굴을 새로 만드는 구조가 아니라, 다음 흐름으로 동작합니다.

1. source avatar 이미지를 전처리해 reusable asset bank로 저장
2. metadata의 `tdmm_raw.coeffs`에서 target 얼굴 landmark 복원
3. source landmark를 target landmark로 piecewise affine warp
4. soft mask, color matching, alpha blending으로 원본 프레임에 합성

## 핵심 아이디어

- source avatar는 런타임마다 다시 분석하지 않고, 미리 `crop / coeff / landmark / mask`를 계산해 둡니다.
- runtime에서는 영상에서 얼굴을 다시 추정하는 대신 metadata에 이미 포함된 `264 coeff`를 그대로 사용합니다.
- 표정은 직접 생성하지 않고, `coeff -> landmark` 복원 결과가 바뀌면서 source avatar의 triangle mesh 목적지 좌표가 바뀌는 방식으로 반영됩니다.
- 얼굴 합성은 landmark 기반 `piecewise affine warp`와 `soft alpha blending` 중심으로 구현되어 있습니다.

## 디렉터리 구조

```text
focus_avatar/
  project/
    models/
      face_landmarker.task
    shared/
      facemap_assets/
        meanFace.npy
        shapeBasis.npy
        blendShape.npy
      converters/
        coeffs_to_landmark.py
    source_avatar/
      source_avatar_prepare.py
      precompute_source_avatar_assets.py
      source_avatar_cli.py
    reenactment/
      reenact_cli.py
      reenact_pipeline.py
      reenact_live.py
      reenact_live_cli.py
      reenact_keyframe_cache.py
      reenact_composite.py
      reenact_assets_runtime.py
      metadata_bbox_utils.py
      reenact_face_planner.py
      reenact_restore.py
```

## 파일별 역할

### `project/source_avatar`

- `source_avatar_prepare.py`
  - source image 전처리용 저수준 유틸
  - MediaPipe 기반 source bbox 검출
  - source crop 생성
  - FaceMap 3DMM coeff 추론
  - coeff를 68개 source landmark로 복원

- `precompute_source_avatar_assets.py`
  - source image 1장에 대해 실제 전처리 파이프라인 수행
  - `source_crop_bgr`, `source_coeff_264`, `source_points`를 `.npz`로 저장

- `source_avatar_cli.py`
  - `front / left / right` source portrait를 읽어 avatar bank 생성
  - 각 view별 `meta.npz`, `mask.png`, `profile.json` 생성

### `project/shared`

- `converters/coeffs_to_landmark.py`
  - FaceMap 3DMM `264 coeff`를 68개 얼굴 landmark로 복원하는 핵심 수학 모듈
  - identity / expression / pose / translation / focal을 분리
  - 평균 얼굴 + basis + 회전/이동 + 2D projection 수행

### `project/reenactment`

- `reenact_cli.py`
  - 전체 reenact 진입점
  - file mode / live mode 선택

- `reenact_pipeline.py`
  - 오프라인 비디오 처리 메인 파이프라인
  - metadata, video, avatar bank를 읽고 최종 output video 생성

- `reenact_live.py`
  - JSONL / stream 기반 실시간 reenact 처리

- `reenact_keyframe_cache.py`
  - frame plan 생성
  - `face_key`, `avatar_id`, `source_view` 결정
  - keyframe warp 결과 캐시 생성

- `metadata_bbox_utils.py`
  - metadata bbox 파싱
  - bbox scale / shift / smoothing
  - `coeff_264` 추출
  - 복원 landmark를 bbox 내부로 fitting

- `reenact_composite.py`
  - source landmark -> target landmark warp
  - Delaunay triangulation 기반 piecewise affine warp
  - soft face mask 생성
  - color matching
  - alpha blending / seamless clone

- `reenact_assets_runtime.py`
  - avatar bank 로드
  - `profile.json`, `meta.npz`, mask 로드
  - yaw 기반 `front / left / right` view 선택

- `reenact_restore.py`
  - 선택적 GPEN face restoration 후처리

## 전처리 파이프라인

source avatar 이미지는 바로 사용하지 않고, 아래 단계로 전처리합니다.

1. source 이미지 로드
2. MediaPipe FaceLandmarker로 얼굴 bbox 검출
3. bbox 기준으로 `256x256` 얼굴 crop 생성
4. FaceMap 3DMM 모델로 `264 coeff` 추론
5. `meanFace.npy`, `shapeBasis.npy`, `blendShape.npy`를 사용해 68개 source landmark 복원
6. landmark convex hull 기반 soft face mask 생성
7. `meta.npz`, `mask.png`, `profile.json`으로 저장

전처리 결과는 보통 다음 형태의 bank가 됩니다.

```text
avatar_bank/<avatar_id>/
  front.png
  front_meta.npz
  front_mask.png
  left.png
  left_meta.npz
  left_mask.png
  right.png
  right_meta.npz
  right_mask.png
  profile.json
```

## Reenact 파이프라인

runtime에서는 source avatar를 다시 분석하지 않고, metadata와 avatar bank를 사용합니다.

1. metadata, video, avatar bank 로드
2. avatar bank에서 사용 가능한 `avatar_id` 수집
3. metadata를 훑어 frame별 얼굴 처리 계획 생성
4. 얼굴마다 `face_key` 생성
5. 얼굴마다 `avatar_id`와 `source_view(front / left / right)` 결정
6. metadata의 `coeff_264`로 target landmark 복원
7. source avatar를 target landmark로 warp
8. soft mask, coverage mask, LAB color matching, alpha blending으로 프레임에 합성

## `coeff -> landmark` 복원 방식

`project/shared/converters/coeffs_to_landmark.py`는 FaceMap 출력 `264 coeff`를 다음처럼 분해합니다.

- `0:219`: identity
- `219:258`: expression
- `258:261`: pitch, yaw, roll
- `261:264`: translation x/y, focal

그 다음:

1. 평균 얼굴(`meanFace`)에 identity basis와 expression basis를 더해 3D 얼굴 형상 생성
2. pitch / yaw / roll 회전 적용
3. translation과 focal 적용
4. 3D 점을 2D landmark로 projection

즉 표정은 expression coeff가 바꾸고, 고개 방향은 pose coeff가 바꾸며, 최종 target landmark가 warp의 목적지 shape가 됩니다.

## Warp 구현 방식

warp는 얼굴 전체에 transform 1개를 거는 방식이 아니라, landmark 기반 triangle mesh를 이용하는 방식입니다.

1. target landmark 기준 Delaunay triangulation 생성
2. triangle마다 source 점 3개와 target 점 3개 대응
3. `cv2.getAffineTransform()`으로 triangle별 affine matrix 계산
4. `cv2.warpAffine()`으로 source patch 변형
5. triangle mask로 유효 영역만 output에 누적
6. coverage mask를 함께 생성

이 방식 덕분에:

- 입 주변
- 눈 주변
- 턱선

같은 얼굴 부위가 서로 다른 방식으로 local deformation 될 수 있습니다.

## `coverage_mask`란?

`coverage_mask`는 piecewise affine warp 과정에서 triangle들이 실제로 채운 영역을 누적한 마스크입니다.

- landmark 기반 face mask: 얼굴일 것 같은 이론적 영역
- `coverage_mask`: warp 결과가 실제로 유효하게 채워진 영역

최종 합성에서는 두 마스크를 결합해서, 실제로 유효한 warp 영역 안쪽만 보수적으로 합성합니다.

## Color Matching / Blending

최종 합성 전에는 target ROI와 source face의 색 차이를 줄이기 위해 LAB color matching을 수행합니다.

- source face와 target ROI를 LAB 색공간으로 변환
- 얼굴 마스크 내부 픽셀 위주로 채널별 평균/표준편차 계산
- source 색 통계를 target 쪽으로 이동
- `strength` 비율만큼만 반영

합성은 기본적으로 soft face mask 기반 alpha blending을 사용합니다.

- 얼굴 중심부는 warp 결과를 강하게 반영
- 경계는 alpha를 낮춰 원본과 부드럽게 섞음

선택적으로 OpenCV `seamlessClone`도 사용할 수 있습니다.

## 멀티페이스 처리 방식

멀티페이스는 각 얼굴을 독립된 reenact 단위로 처리하는 구조입니다.

- 프레임 안 얼굴들을 bbox 큰 순서대로 정렬
- 얼굴마다 `face_key` 생성
- 얼굴마다 `avatar_id` 배정
- 얼굴마다 `source_view` 선택
- 얼굴마다 독립 warp 계산
- 같은 프레임 위에 순차적으로 composite

같은 사람에 같은 아바타가 유지되도록 `face_key -> avatar_id` 매핑을 유지합니다.

## 실행 예시

아래 예시는 현재 구조 기준의 대표적인 실행 방식입니다.

### 1. source avatar bank 생성

`project` 디렉터리에서 실행:

```bash
cd /Users/yunsol/cap_co/focus_avatar/project
python -m source_avatar.source_avatar_cli \
  --source-dir /path/to/source_avatar_dir \
  --output-dir /path/to/avatar_bank
```

### 2. 오프라인 reenact 실행

```bash
cd /Users/yunsol/cap_co/focus_avatar/project
python -m reenactment.reenact_cli \
  --run-mode file \
  --metadata /path/to/metadata.json \
  --video /path/to/input.mp4 \
  --output-video /path/to/output.mp4 \
  --avatar-bank-dir /path/to/avatar_bank
```

### 3. live reenact 실행

```bash
cd /Users/yunsol/cap_co/focus_avatar/project
python -m reenactment.reenact_cli \
  --run-mode live \
  --metadata /path/to/metadata.jsonl \
  --video /path/to/input.mp4 \
  --output-video /path/to/output.mp4 \
  --avatar-bank-dir /path/to/avatar_bank
```

## 필요 자산 / 의존성

주요 의존성:

- Python 3
- `opencv-python`
- `numpy`
- `torch`
- `scipy`
- `mediapipe`
- `qai-hub-models`
- 선택: `onnxruntime`

필수 자산:

- `project/models/face_landmarker.task`
- `project/shared/facemap_assets/meanFace.npy`
- `project/shared/facemap_assets/shapeBasis.npy`
- `project/shared/facemap_assets/blendShape.npy`

## 한계

- source 이미지에 없는 디테일은 새로 생성하지 못합니다.
- 치아, 혀, 극단적인 pose 변화, 심한 가림은 triangle warp만으로 완벽히 해결되지 않습니다.
- 멀티페이스 겹침이 심한 경우 depth 기반 occlusion reasoning은 없습니다.

## 한 줄 요약

`focus_avatar`는 **전처리된 source avatar bank와 metadata의 3DMM coefficient를 이용해 target landmark를 복원하고, source avatar를 piecewise affine warp와 soft blending으로 원본 영상에 합성하는 시스템**입니다.
