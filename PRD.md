## Product Requirements Document (PRD)

### 1. Project Overview

본 프로젝트는 노트북 웹캠과 MediaPipe 기반 손 추적으로 NASA 우주인 3D 모델을 실시간 조작하는 인터랙티브 데스크톱 앱이다.  
현재 구현 목표는 "두 손으로 물체를 잡아 움직이는 감각"을 제공하면서, 손 인식 불안정 구간에서도 조작이 튀지 않도록 안정성을 확보하는 것이다.

### 2. Target Users

- 우주/3D 인터랙션 데모를 빠르게 만들고 싶은 개발자
- Computer Vision + 3D 제어 매핑을 학습하는 사용자
- 비접촉식 인터페이스(NUI) 프로토타입이 필요한 기획/연구 사용자

### 3. Product Goals

- 별도 센서 없이 웹캠만으로 3D 우주인 조작
- 1손/2손 제스처를 분리해 직관적인 모드 제공
- 모델/라이브러리 호환성 이슈(텍스처, API 버전) 흡수
- 실시간 HUD로 추적 상태를 즉시 피드백

### 4. Scope (Current Implementation)

#### F1. Hand Tracking Engine

- 최대 2손, 21 랜드마크 추적
- MediaPipe 구버전 `solutions`와 신버전 `tasks` 백엔드 자동 대응
- `tasks` 환경에서 `models/hand_landmarker.task` 자동 다운로드
- 손 미검출 시 상태를 `Hand Lost`로 표기

#### F2. 3D Scene and Model Rendering

- Ursina/Panda3D 기반 렌더링
- 검정 배경 + 스타필드(우주 배경)
- 우주인 모델 로드 우선순위:
  - `Astronaut_plain.obj`
  - `Astronaut_compat.glb`
  - `Astronaut_plain.glb`
  - `Astronaut.glb`
  - `Astronaut_converted.obj`
- 텍스처/호환성 처리:
  - Draco/WebP 호환본 생성 지원
  - 호환 셰이더 적용
  - 양면 렌더링(`setTwoSided`)으로 모델 면 누락(뚫림) 완화

#### F3. Gesture Interaction

- `1손 모드`: 회전 전용
  - 검지 이동으로 회전
  - 손목 비틀기로 추가 회전
  - 현재 회전 방향은 손 동작의 역방향으로 매핑
- `2손 모드`: 이동 + 줌 + 회전
  - 양손 중심 이동 -> 우주인 이동
  - 손 간 거리 변화 -> 줌(가까우면 확대, 멀어지면 축소)
  - 양손 상대 각도/롤 -> 회전
  - 손이 매우 가까운 구간에서는 `zoom-only` 잠금으로 회전 튐 방지
- 스무딩(lerp) 적용으로 급격한 흔들림 완화

#### F4. HUD / UX Feedback

- 우하단 웹캠 미리보기 패널
- 좌상단 FPS, Gesture 상태 표시
- 손 미검출 시 경고 텍스트 표시
- `ESC`로 종료

### 5. Functional Requirements

1. 앱 실행 시 3D 씬과 웹캠 캡처를 초기화한다.
2. 손 추적 결과(1손/2손)에 따라 모드를 자동 전환한다.
3. 2손 줌 시 가까운 거리에서 회전이 발생하지 않도록 안정화 로직을 적용한다.
4. 모델 파일/텍스처 포맷 호환성 문제 발생 시 폴백 경로를 통해 렌더링을 유지한다.
5. 손이 사라지면 제스처 상태를 `Hand Lost`로 표기하고 앱은 중단되지 않는다.

### 6. Non-Functional Requirements

- 실시간 인터랙션 체감(환경에 따라 약 20 FPS 이상 권장)
- macOS 카메라 권한 미허용 시 에러 메시지로 원인 노출
- Python 3.13 환경에서도 MediaPipe `tasks` 경로로 동작 가능

### 7. Tech Stack

- Language: Python
- CV/Tracking: OpenCV, MediaPipe
- 3D Engine: Ursina (Panda3D)
- Model Source: NASA 3D Resources (Astronaut asset)

### 8. Acceptance Criteria (MVP)

1. 앱 실행 후 우주인 모델이 화면 중앙에 렌더링된다.
2. 1손에서 회전이 동작하고, 2손에서 이동/줌/회전이 동작한다.
3. 손을 매우 가깝게 모아 확대할 때 회전이 급격히 튀지 않는다.
4. HUD에 FPS/제스처/손 상태가 정상 표시된다.
5. `mediapipe` 버전에 따라 `solutions` 또는 `tasks` 중 하나로 자동 동작한다.

### 9. Out of Scope (Current)

- 정식 제스처 사전(예: 손모양 분류 기반 명령셋) 구축
- 음성/키보드/게임패드 복합 입력
- 멀티 오브젝트 씬 편집 기능
- 배포 패키징(.app/.exe) 자동화

### 10. Next Iterations

- 제스처 감도 프리셋(민감도/반전/줌 한계) UI 제공
- 환경 조명/톤매핑 조절로 모델 시인성 개선
- 기록/리플레이(제스처 입력 로그) 기능
