## 📄 Product Requirements Document (PRD)

### 1. Project Overview

본 프로젝트는 별도의 하드웨어 센서 없이 **노트북 웹캠과 AI(MediaPipe)**를 활용하여 NASA의 3D 우주인 모델을 실시간으로 조작하는 인터랙티브 애플리케이션이다. 사용자의 손동작을 분석하여 우주인의 위치, 회전, 크기를 제어함으로써 마치 우주 공간에서 우주인을 조종하는 듯한 경험을 제공한다.

### 2. User Target

* 우주 및 공학 매니아
* 컴퓨터 비전 및 3D 그래픽스 학습자
* 비접촉식 인터페이스(NUI) 프로토타입 개발자

### 3. Key Features (주요 기능)

#### **F1. 실시간 손가락 추적 (Hand Tracking)**

* MediaPipe Hands를 이용해 21개의 손마디 좌표를 실시간으로 추출.
* 최소 20 FPS 이상의 처리 속도 유지.

#### **F2. 3D 우주인 렌더링**

* NASA 3D Resources의 `Astronaut.glb` 모델 로딩.
* 우주 공간 느낌의 배경(Starfield) 및 조명 설정.

#### **F3. 제스처 매핑 제어 (Core Interaction)**

* **Rotation (회전)**: 검지 손가락의 움직임에 따라 우주인의 축 회전.
* **Zoom (확대/축소)**: 엄지와 검지 사이의 거리를 계산하여 모델의 스케일 조절.
* **Translation (이동)**: 손바닥 전체의 위치에 따라 우주인이 화면 내에서 이동.

#### **F4. 시각적 피드백 (HUD)**

* 화면 구석에 웹캠 피드백 창 출력 (손 추적 상태 확인용).
* 현재 프레임(FPS) 및 감지된 제스처 상태 표시.

### 4. Technical Stack

* **Language**: Python 3.9+
* **AI Engine**: Google MediaPipe (Hand Landmarking)
* **3D Engine**: Ursina Engine (Panda3D 기반, 쉽고 빠른 프로토타이핑 가능)
* **Computer Vision**: OpenCV
* **Model Source**: NASA 3D Resources (Astronaut.glb)

---

### 5. Functional Requirements (상세 요구사항)

1. **초기화**: 프로그램 실행 시 웹캠을 활성화하고 3D 윈도우를 생성한다.
2. **모델 로드**: 지정된 경로의 `.glb` 파일을 텍스처와 함께 정확히 렌더링한다.
3. **좌표 정규화**: 카메라 해상도와 3D 공간 좌표계를 매핑하여 손의 움직임이 모델에 자연스럽게 반영되도록 한다.
4. **예외 처리**: 손이 화면 밖으로 나갔을 때 모델의 마지막 상태를 유지하고 "Hand Lost" 메시지를 표시한다.

### 6. Roadmap (개발 단계)

* **Phase 1**: Python 환경 구축 및 MediaPipe 손 인식 기본 코드 작성.
* **Phase 2**: Ursina Engine으로 NASA 우주인 모델 띄우기.
* **Phase 3**: 손 좌표 데이터를 모델의 변수(Rotation, Scale)에 연결.
* **Phase 4**: 조작감 튜닝(Smoothing 필터 적용) 및 배경 꾸미기.

---