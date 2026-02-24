# astro-gesture-3d

웹캠 + MediaPipe + Ursina로 3D 우주인(`Astronaut.glb`)을 손동작으로 조작하는 인터랙티브 앱입니다.

## Features

- 실시간 손가락 추적(21개 랜드마크, 최대 2손)
- 3D 우주인 모델 렌더링 + 우주 배경(스타필드)
- 모델 로딩 정책:
  - `Hubble Space Telescope (B).glb`가 있으면 허블 모델만 로딩(실패 시 에러, 우주인 폴백 없음)
  - 허블 파일이 없을 때만 `Astronaut_plain.obj` / `Astronaut_compat.glb` / `Astronaut_plain.glb` / `Astronaut.glb` / `Astronaut_converted.obj` 순으로 폴백
- MediaPipe 구버전(`mp.solutions`)과 신버전(`tasks`) 모두 지원
- 제스처 매핑
  - `1손`: 회전 전용 (검지 이동)
  - `2손`: 이동 + 줌 + 회전 (손쌍 중심 이동 + 손 사이 거리 + 양손 회전)
- HUD
  - 우하단 웹캠 피드
  - 좌상단 FPS / Gesture 상태
  - 손 미검출 시 `Hand Lost`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 app.py
```

## 손으로 조종하기

1. 앱 실행 후 카메라 권한을 허용합니다.
2. 화면 우하단 웹캠 패널에 손(1손 또는 2손)이 보이도록 손바닥을 카메라 정면으로 듭니다.
3. 손이 인식되면 좌상단 `Gesture`가 `Tracking` 또는 제스처 이름으로 바뀝니다.
4. 아래 동작으로 우주인을 조종합니다.

- `1손 모드 (HUD에 1H 표시)`
  - `Rotation`: 검지 손끝 이동으로 회전합니다.
  - `Rotation (Twist)`: 손 자체를 돌리면(손목 비틀기) 같은 방향으로 추가 회전합니다.
- `2손 모드 (HUD에 2H 표시)`
  - `Translation`: 두 손으로 우주인을 잡고 끌듯이, 양손 중심을 움직이면 우주인이 함께 이동합니다.
  - `Zoom`: 두 손 사이 거리가 가까워지면 확대되고, 멀어지면 축소(멀어짐)됩니다.
  - `Rotation`: 양손을 비틀어(손목 롤) 돌리면 우주인이 함께 회전합니다. 두 손의 상대 각도(손쌍 방향)를 바꾸면 추가 회전이 들어갑니다.

### 인식이 잘 안 될 때

- 손 전체(손가락 포함)가 프레임 안에 들어오게 유지하세요.
- 배경과 손 색이 너무 비슷하면 추적이 흔들릴 수 있습니다.
- 너무 빠른 손동작은 건너뛰는 프레임이 생길 수 있어, 처음엔 천천히 움직여 보세요.
- `Hand Lost`가 뜨면 손을 다시 화면 중앙으로 가져오면 됩니다.

## Controls

- `ESC`: 종료
- `R`: 모델 위치/회전/줌 즉시 초기화(화면 중앙 복귀)

## Notes

- macOS에서는 카메라 접근 권한 허용이 필요합니다.
- `Hubble Space Telescope (B).glb`를 프로젝트 루트에 두면 허블 망원경 모델이 우선 로딩됩니다.
- 허블 모델이 Draco 압축본이면 실행 시 `Hubble Space Telescope (B)_plain.glb`가 자동 생성되어 사용됩니다.
- `Astronaut.glb` 파일은 프로젝트 루트에 있으면 폴백 모델로 사용됩니다.
- `Astronaut_plain.glb`는 Draco 압축 해제된 호환본 모델입니다.
- `Astronaut_plain.obj`는 Panda3D에서 텍스처 인식 안정성이 높은 폴백 모델입니다.
- `Astronaut_compat.glb`는 WebP 텍스처 확장을 PNG 텍스처로 치환한 렌더링 호환본입니다.
- `Astronaut_converted.obj`, `material.mtl`, `astnt1_1.png`, `astnt1_2.png`는 GLB 폴백 렌더링용입니다.
- `mediapipe`가 `tasks` API만 제공하는 환경(Python 3.13 등)에서는 첫 실행 시 `models/hand_landmarker.task`를 자동 다운로드합니다.

## Troubleshooting

- 화면이 검정색으로만 보일 때:
  - 먼저 `R` 키로 모델을 중앙에 리셋해 보세요.
  - 손을 모두 내리고 1~2초 기다리면 모델이 자동으로 홈 위치로 복귀합니다.
  - 콘솔에 `OpenCV: camera failed to properly initialize!`가 나오면 macOS 카메라 권한을 허용한 뒤 앱을 재실행하세요.
