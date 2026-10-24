# HRM body segmentation

검은 배경에서 촬영한 RGB와 기존 영상처리 마스크를 이용하는 **binary semantic segmentation** 프로젝트입니다.
ImageNet 사전학습 **U-Net / ResNet18**을 기본으로 사용합니다. 원본 크기에 가까운 384 입력,
작은 decoder, CUDA mixed precision으로 속도와 경계 정확도를 함께 고려했습니다.
실시간 여부는 사용하는 GPU에서 `benchmark` 명령으로 확인하세요.

## 데이터에서 확인한 내용

- `datasets/image_rgb` / `datasets/image_segmentation`: 이름이 일치하는 **5,389쌍**.
- RGB: `337×379×3`, uint8. 마스크: `337×379`, uint8, **0과 255만 존재**.
- 마스크의 흰색은 **흰 몸체 전체**입니다. 검은 기저부·끝단은 제외되며 개별 관절 번호는 없습니다.
- 저장 형식 `0/255`는 정상입니다. 학습 직전에 `(mask > 0)`으로 **float32 0/1**로 변환합니다.
  모델 출력은 sigmoid 확률이며 기본 임계값 0.5로 이진화하고, 결과 PNG는 다시 `0/255`로 저장합니다.
- 전경은 이미지의 약 3.1–5.9%입니다. 손실함수는 `0.5 BCE + 0.5 Dice`입니다.

현재 세션은 파일명의 초·나노초 timestamp 사이 2초를 넘는 간격으로 구분됩니다.
기본적으로 두 세션 3,860장을 학습하고, 마지막 세션 1,529장을 검증합니다.
인접 프레임을 무작위로 섞지 않습니다. **독립적인 실제 수술 환경 테스트 세트는 아직 없습니다.**
검증 성능으로 체크포인트를 선택하므로 검증 결과를 독립 테스트 성능으로 해석하면 안 됩니다.
중첩 원본/legacy 폴더의 이미지는 자동으로 추가하지 않습니다.

## 설치

Python 3.10 이상을 사용합니다. RTX 3060과 RTX 5060 Ti를 고려해 CUDA 12.8 PyTorch wheel을 사용합니다.
드라이버와 Python에 맞는 다른 설치가 필요하면 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/previous-versions/)를 확인하세요.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

이 작업 공간에는 `.venv` 환경을 구성했습니다. `uv`를 사용한다면 같은 명령을
`uv pip install --python .venv/bin/python ...` 형태로 실행할 수 있습니다.

## 실행 순서

```bash
source .venv/bin/activate

# 전체 마스크 검사 + 촬영 세션별 분리. 원본 이미지는 변경하지 않습니다.
python hrm.py prepare

# 공식 공개 수술 영상에서 배경 300장 확보. 기존 30장도 보존합니다.
python backgrounds.py --train 180 --val 60 --test 60

# 무작위 배치만 / 배치와 외관 열화를 함께 적용한 모습을 확인합니다.
python hrm.py preview --mode placement --output runs/placement_random_preview.jpg
python hrm.py preview --mode combined --output runs/combined_random_preview.jpg

# 기본값: 원본 50장 × 2개 모드 × 3개 변형 = RGB/마스크 300쌍 저장
python hrm.py export-augmented

# RTX 4090 예시. 3060에서는 batch-size 8부터 시작하고 부족하면 4로 줄이세요.
python hrm.py train --epochs 20 --batch-size 16 --size 384 --output runs/hrm_random_placement

# 위 새 학습을 마친 뒤 같은 검증 세션에서 각 조건을 별도로 평가합니다.
python hrm.py evaluate --checkpoint runs/hrm_random_placement/best.pt \
  --output runs/hrm_random_placement/evaluation.json

# 사진, 이미지 폴더, 영상 또는 웹캠 번호를 입력할 수 있습니다.
python hrm.py predict --checkpoint runs/hrm_random_placement/best.pt --source datasets/image_rgb
python hrm.py predict --checkpoint runs/hrm_random_placement/best.pt --source 0 --show
```

`prepare`는 기존 `datasets/split.json`을 덮어쓰지 않습니다. 의도적으로 분리를 바꿀 때만
`--overwrite`를 사용하세요. 학습한 파일 목록을 checkpoint에 보관하여 평가 시 학습 데이터와의 겹침도 검사합니다.
`--val-group session_03 --test-group session_02`처럼 독립 촬영 세션을 테스트로 분리할 수도 있지만,
현재 세 세션으로 그렇게 하면 학습 세션이 하나만 남습니다.

`train`은 기존 실행 결과를 덮어쓰지 않습니다. 재실험은 다른 `--output`을 지정합니다.
`best.pt`는 원본 검증 Dice와 고정 합성 검증 256장의 Dice 평균이 가장 높은 모델이며,
`last.pt`, `history.csv`, `config.json`, 분리/배경 출처 사본도 저장합니다.
학습은 매번 HRM의 위치·방향과 외관을 새로 증강하므로 합성 이미지 수만 장을 미리 저장할 필요가 없습니다.

`export-augmented`는 확인용 RGB와 `0/255` 마스크를 `datasets/augmented_random`의
`image_rgb/`, `image_segmentation/`에 저장하고 `manifest.json`, 배경 출처·라이선스 사본,
`preview.jpg`를 만듭니다. `--count`, `--variants`, `--modes`, `--split`, `--output`으로 바꿀 수 있습니다.
기존 출력 폴더는 덮어쓰지 않으므로 다시 내보낼 때는 새 `--output`을 지정하세요.
이 파일들은 학습 당시 배치를 복원한 것이 아니라, seed가 기록된 증강 예시입니다.
기존 `datasets/augmented_samples`의 60쌍은 이전 30장 배경·제한된 자세 조건의 예시로 보존합니다.
[저장된 300쌍 미리보기](datasets/augmented_random/preview.jpg)와
[원본 / 위치 증강 / 정답 비교](runs/placement_preview.jpg)를 확인할 수 있습니다.

사전학습 encoder는 첫 학습 시 다운로드됩니다. `--no-pretrained`는 오프라인 동작 확인용입니다.

정확도 비교용으로 `--encoder resnet34 --size 512`를 사용할 수 있습니다.
성능 개선이 보장되는 설정은 아니며 VRAM·처리 시간을 함께 측정해야 합니다.
기본 U-Net은 [segmentation-models-pytorch](https://smp.readthedocs.io/en/latest/models.html)의 검증된 구현을 사용합니다.
현재 라벨은 몸체 전체의 픽셀 마스크여서 YOLO용 객체 polygon으로 변환하지 않습니다.

## 배경과 증강

`backgrounds.py`는 [공식 Endoscapes 데이터](https://github.com/CAMMA-public/Endoscapes)에서
현재 **300장, 원본 영상 201개**를 확보했습니다. 학습 180장/120영상, 검증 60장/41영상,
테스트 60장/40영상이며 공식 영상별 분리를 유지합니다. 기존 30장을 보존하고 새 영상부터 채운 뒤,
같은 영상에서 시간 간격이 큰 프레임을 추가합니다. 모두 복강경 담낭절제술 영상으로,
다른 술식이나 장기 영역까지 확보한 데이터는 아닙니다.
대형 ZIP을 전부 받지 않고 HTTP Range로 목록과 선택한 JPEG만 추출합니다.
서버가 Range를 지원하지 않으면 전체 다운로드로 전환하지 않고 실패합니다.
완료 후에는 저장된 SHA256과 이미지 디코딩·비검정 여부를 확인하고 재다운로드를 생략합니다.
`--train`, `--val`, `--test`는 최소 목표 장수이며 작은 값을 입력해도 기존 파일을 삭제하지 않습니다.
확장 전 메타데이터는 `datasets/backgrounds/manifests/`에 보관합니다.
[배경 미리보기](datasets/backgrounds/preview.jpg)는 60개 영상의 대표 장면입니다.

`datasets/backgrounds/{train,val,test}`에 직접 확보한 배경도 넣을 수 있습니다.
동일 배경 영상의 프레임을 여러 split에 섞지 마세요. 직접 추가한 배경의 출처와 이용 조건도 따로 기록하세요.
해당 split의 배경이 없으면 메시지를 출력하고 일반적인 랜덤 색상/질감을 사용합니다.
이 대체 배경은 수술 영상이 아닙니다.

원본 Endoscapes와 이를 이용한 파생 데이터에는 **CC BY-NC-SA 4.0** 조건이 적용됩니다.
비상업적 이용, 출처 표시, 파생물의 동일조건 공유를 준수해야 하며 저장소의 코드 라이선스와 구분됩니다.
정확한 원본·라이선스·논문 정보는 `datasets/backgrounds/sources.json`과 라이선스 파일을 참고하세요.

| 증강 | 구현 및 정답 처리 |
|---|---|
| 배경 합성 | 기존 몸체 마스크 안의 RGB를 보존하고 배경 교체. 경계 안쪽으로만 feathering |
| 위치·방향 | 몸체를 분리해 화면 안 임의 위치에 배치하고 0–360° 회전. 요청 배율 0.65–1.35를 회전 후 화면 크기에 맞게 제한. 마스크도 같은 좌표로 변환 |
| 몸체 색 | 표면 명암을 유지하는 RGB 배율 변화 |
| 조명 | 감마·밝기·색온도·공간적 그림자·밝은 반사 영역 |
| 연기 | 불균일한 반투명 연기층. 객체 정답 마스크 유지 |
| 가림 | 불투명한 불규칙 영역. 가려진 픽셀은 정답에서도 제거 |
| 카메라 | Gaussian blur, 센서 노이즈, JPEG 열화 |
| 배경만 있는 장면 | 학습의 약 5%. 정답 전체가 0인 사례로 오검출 억제 |

`placement`는 몸체의 위치·회전·크기와 배경만 바꿉니다. 배치의 약 70%는 몸체 전체가 보이며,
약 30%는 화면 가장자리에서 일부 잘리도록 시도합니다. 잘릴 때도 변환된 전경 면적의 **60% 이상**을
남기며, 이 기준은 이후 불투명 가림을 적용하기 전의 값입니다. 마스크는 nearest-neighbor로 변환하고
RGB는 마스크를 곱한 상태로 보간하여 원본 검은 배경이 경계에 섞이지 않게 합니다.
`combined`는 이 배치에 색·조명·연기·가림·카메라 열화를 더합니다.

학습에는 무작위 배치, 약한 조건, 복합 조건을 섞고 일부 원래 외관도 유지합니다.
`preview --mode placement`, `--mode combined`, `--mode smoke`, `--mode occlusion` 등으로 각각 확인할 수 있습니다.
검증용 증강은 파일명과 seed로 고정되며 학습용 배경을 사용하지 않습니다.

연기·가림은 초기 강건성 실험용 근사입니다. 실제 수술 연기·조명·조직 접촉을 물리적으로 재현한 것은 아닙니다.
300장은 한 술식에서 고른 표본이며, 합성만으로 실제 수술 환경의 성능을 보장하지 않습니다.
2D 증강은 새로운 관절 형상이나 3D 촬영 시점을 생성하지 않습니다. 실제 목표 환경에서의 추가 촬영과 검증이 필요합니다.
영상처리로 생성한 정답의 경계 오류도 모델에 학습될 수 있으므로 대표 마스크를 검수하세요.

## 평가와 속도 측정

```bash
python hrm.py evaluate --checkpoint runs/hrm_random_placement/best.pt --limit 300 \
  --output runs/hrm_random_placement/evaluation_quick.json
python hrm.py benchmark --checkpoint runs/hrm_random_placement/best.pt \
  --source datasets/image_rgb/00000001_1790484638_813745361.png
python -m unittest -v test_hrm.py
```

평가 결과는 Dice, IoU, precision/recall, 입력 해상도에서 2px 허용 경계 F1을 기록합니다.
`negative` 조건에서는 정답 전경이 없으므로 Dice보다는 `predicted_foreground_fraction`과
`false_positive_pixels_per_image`를 보세요. 평가와 benchmark는 checkpoint의 입력 크기를 자동으로 사용합니다.

benchmark는 batch 1, warmup 후 CUDA 동기화를 포함해 측정합니다.
`model`은 GPU에 이미 올라간 텐서의 모델 실행,
`pipeline`은 CPU 전처리·CPU/GPU 전송·모델·원래 해상도 복구·이진화를 포함합니다.
카메라 수신·영상 디코딩·overlay·화면 표시·파일 저장 시간은 제외됩니다.
**4090에서 측정한 결과를 3060/5060 Ti의 FPS로 해석하지 마세요.**

사진 추론은 원래 크기의 `*_mask.png`와 `*_overlay.jpg`를 저장합니다.
영상/웹캠은 `overlay.mp4`를 저장하고, `--save-masks`로 프레임별 예측 마스크 PNG도 저장합니다.
화면이 없는 서버에서는 `--show`를 생략합니다.

## 확장된 배경·위치 증강으로 확인한 결과 — 2026-10-21

배경 300장과 무작위 배치를 적용해 RTX 4090에서 20 epoch 학습을 완료했습니다.
새 모델은 `runs/hrm_random_placement/best.pt`이며 최적 checkpoint는 epoch 18입니다.
기존 15 epoch 모델 `runs/hrm/best.pt`도 보존했습니다. 위 학습 명령을 다시 실행하려면
`--output runs/hrm_random_placement_v2`처럼 새로운 출력 경로를 사용하세요.

두 모델에 **동일한 검증 원본 256장, seed, 확장된 검증 배경 60장, 증강 코드**를 적용했습니다.
아래는 새 조건에서 비교한 Dice이며, 배경 확대와 위치 증강 및 추가 학습을 함께 적용한 결과입니다.

| 동일한 검증 조건 | 기존 모델 | 새 모델 |
|---|---:|---:|
| 원본 | 99.54% | 99.42% |
| 임의 위치·360° 회전·배경 합성 | 65.77% | 98.79% |
| 위 배치 + 색·조명·연기·가림·카메라 열화 | 63.89% | 97.94% |

배경만 있는 256장에서는 이미지당 오검출 픽셀이 36.82개에서 10.50개로 줄었습니다
(384 입력 좌표 기준). 수치는 `runs/hrm_random_placement/comparison.json`과
`evaluation_256.json`에 있고, [기존 / 새 모델 예측 비교](runs/hrm_random_placement/prediction_examples.jpg)도 저장했습니다.
이 검증 세션은 checkpoint 선택에도 사용했으므로 독립 테스트 성능이 아니며, 실제 수술 중 HRM 영상 성능도 아닙니다.

300쌍 전체의 RGB·마스크 크기와 `0/255` 값, 학습 split 일치, 사분면·화면 경계 분포를 검사했습니다.
3쌍은 기록된 seed로 다시 생성해 PNG 바이트까지 일치하는 것을 확인했습니다.
결과는 `datasets/augmented_random/audit.json`에 있습니다. 학습 폴더에는 증강 코드 사본,
배경 목록·해시, 데이터 분리와 학습 로그도 보관합니다.

## 이전 조건에서 확인한 결과 — 2026-10-21

아래 모델·표·예시는 **배경 30장과 제한적인 이동·회전으로 학습·평가한 기존 결과**입니다.
확장된 배경 300장과 360° 무작위 배치의 성능을 뜻하지 않습니다. 새 조건의 학습·평가 결과는 별도로 기록합니다.

ResNet18 U-Net을 `384×384`, batch 16, 15 epoch 학습했습니다. 최적 checkpoint는 epoch 14의
`runs/hrm/best.pt`이며 바로 추론에 사용할 수 있습니다. 다시 학습하려면 기존 결과를 보존하도록
`--output runs/hrm_random_placement_v2` 등 새 경로를 지정하세요.

검증 세션 1,529장 각각에 같은 seed의 조건을 적용한 결과입니다. 합성 검증 배경은 별도 영상 5개입니다.

| 검증 조건 | Dice | IoU |
|---|---:|---:|
| 원본 | 99.54% | 99.08% |
| 배경 합성 | 99.71% | 99.43% |
| 조명 | 99.42% | 98.85% |
| 연기 | 99.49% | 98.98% |
| 가림 | 99.16% | 98.33% |
| 색 변화 | 99.44% | 98.89% |
| 복합 증강 | 98.90% | 97.82% |

배경만 있는 합성 검증에서는 전경으로 잘못 예측한 픽셀이 0개였습니다.
이는 작은 공개 배경 표본에서의 결과이며 일반적인 수술 장면의 오검출률을 뜻하지 않습니다.
전체 수치는 `runs/hrm/evaluation.json`, 환경 버전은 `runs/hrm/environment.txt`에 있습니다.

실제 연결된 **RTX 4090**에서 batch 1, AMP, warmup 20회 후 200회 측정했습니다.
모델 실행 중앙값 **1.19ms**, 전처리·전송·후처리 포함 중앙값 **2.93ms**,
평균 처리율 **335 FPS**였습니다. 카메라/디코딩/overlay/저장/표시는 제외합니다.
3060과 5060 Ti에서는 직접 측정하지 않았습니다. 결과는 `runs/hrm/benchmark.json`에 있습니다.

현재 회귀 검사는 **11개**이며 마스크·가림 정합성, 좌표 복원, 손실함수, 데이터 누출 방지와
무작위 배치의 정답 정합성을 다룹니다. 기존 체크포인트의 사진·영상 추론과 마스크 저장도 확인했습니다.

- [이전 조건의 증강 예시](runs/augmentation_preview.jpg)
- [입력 / 정답 / 예측 비교](runs/hrm/prediction_examples.jpg)

## 파일 구성

- `hrm.py`: 데이터 검사·분리, 증강 예시 내보내기, U-Net 학습·평가·추론·벤치마크 CLI
- `augment.py`: NumPy/OpenCV 기반 이미지와 정답을 함께 처리하는 증강
- `backgrounds.py`: 공개 배경 다운로드·확장, 출처 기록과 미리보기
- `test_hrm.py`: 마스크/가림/좌표 복구/손실함수/데이터 누출 회귀 검사
- `requirements.txt`: 의존성

데이터, 가상환경, 모델, 실험 출력은 `.gitignore`로 제외합니다.
