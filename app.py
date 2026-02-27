from __future__ import annotations

import atexit
import builtins
import random
import shutil
import subprocess
import time as pytime
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from panda3d.core import SamplerState
from panda3d.core import Shader
from panda3d.core import Texture as PandaTexture
from panda3d.core import CullFaceAttrib
from panda3d.core import TransparencyAttrib
from ursina import (
    Entity,
    Text,
    Texture,
    Ursina,
    Vec2,
    Vec3,
    application,
    camera,
    color,
    lerp,
    load_model,
    scene,
    time,
    window,
)


@dataclass
class HandMetrics:
    landmarks: list[tuple[float, float, float]]
    palm_center: tuple[float, float]
    index_tip: tuple[float, float]
    pinch_distance: float
    handedness: Optional[str] = None


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)
HAND_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


_LEGACY_TEXTURE_SHADER: Optional[Shader] = None
HUBBLE_VISUAL_ALPHA = 0.68
HUBBLE_VISUAL_TINT = (0.74, 0.86, 1.0)
HUBBLE_GLOW_COLOR = (0.32, 0.64, 1.0)
HUBBLE_GLOW_ALPHA = 0.34
HUBBLE_GLOW_SCALE = 1.06
SHOW_WEBCAM_PANEL = False


def get_legacy_texture_shader() -> Shader:
    global _LEGACY_TEXTURE_SHADER
    if _LEGACY_TEXTURE_SHADER is None:
        vert = """
varying vec2 uv;
void main() {
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    uv = gl_MultiTexCoord0.xy;
}
"""
        frag = """
uniform sampler2D p3d_Texture0;
varying vec2 uv;
void main() {
    gl_FragColor = texture2D(p3d_Texture0, uv);
}
"""
        _LEGACY_TEXTURE_SHADER = Shader.make(Shader.SL_GLSL, vert, frag)
    return _LEGACY_TEXTURE_SHADER


class HandTracker:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam open failed. Check camera permissions and device status.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.fps = 0.0
        self._last_ts = pytime.perf_counter()
        self._closed = False
        self._task_timestamp_ms = 0

        self.backend: str
        self.mp_hands = None
        self.mp_draw = None
        self.hands = None
        self.hand_landmarker = None
        self._setup_backend()

    def _setup_backend(self) -> None:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.backend = "solutions"
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            return

        self.backend = "tasks"
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = self._ensure_task_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _ensure_task_model() -> Path:
        model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
        if model_path.exists():
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(HAND_LANDMARKER_TASK_URL, timeout=30) as response:
                model_path.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(f"Unable to download MediaPipe model: {HAND_LANDMARKER_TASK_URL}") from exc

        return model_path

    @staticmethod
    def _build_hand_metrics(
        landmarks: list[tuple[float, float, float]], handedness: Optional[str] = None
    ) -> HandMetrics:
        palm_ids = [0, 5, 9, 13, 17]
        palm_x = float(np.mean([landmarks[i][0] for i in palm_ids]))
        palm_y = float(np.mean([landmarks[i][1] for i in palm_ids]))
        index_tip = (landmarks[8][0], landmarks[8][1])
        thumb_tip = (landmarks[4][0], landmarks[4][1])
        pinch_distance = float(np.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1]))
        return HandMetrics(
            landmarks=landmarks,
            palm_center=(palm_x, palm_y),
            index_tip=index_tip,
            pinch_distance=pinch_distance,
            handedness=handedness,
        )

    @staticmethod
    def _draw_landmarks_cv(
        frame: np.ndarray,
        landmarks: list[tuple[float, float, float]],
        hand_label: Optional[str] = None,
        line_color: tuple[int, int, int] = (85, 255, 120),
    ) -> None:
        height, width = frame.shape[:2]
        points = [(int(x * width), int(y * height)) for x, y, _ in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], line_color, 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 3, (60, 170, 255), -1)
        if hand_label:
            wx, wy = points[0]
            cv2.putText(
                frame,
                hand_label,
                (wx + 8, wy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                line_color,
                2,
                cv2.LINE_AA,
            )

    def read(self) -> tuple[list[HandMetrics], np.ndarray, float]:
        ok, frame = self.cap.read()
        if not ok:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return [], blank, self.fps

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_metrics: list[HandMetrics] = []
        if self.backend == "solutions" and self.hands is not None:
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks:
                handedness_items = results.multi_handedness or []
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                    hand_label = None
                    if i < len(handedness_items) and handedness_items[i].classification:
                        hand_label = handedness_items[i].classification[0].label

                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    hands_metrics.append(self._build_hand_metrics(landmarks, hand_label))
                    if self.mp_draw is not None and self.mp_hands is not None:
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        if hand_label:
                            h, w = frame.shape[:2]
                            wrist = hand_landmarks.landmark[0]
                            cv2.putText(
                                frame,
                                hand_label,
                                (int(wrist.x * w) + 8, int(wrist.y * h) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (120, 250, 120),
                                2,
                                cv2.LINE_AA,
                            )
        elif self.hand_landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(pytime.perf_counter() * 1000)
            if ts <= self._task_timestamp_ms:
                ts = self._task_timestamp_ms + 1
            self._task_timestamp_ms = ts

            task_result = self.hand_landmarker.detect_for_video(mp_image, ts)
            if task_result.hand_landmarks:
                for i, hand in enumerate(task_result.hand_landmarks[:2]):
                    hand_label = None
                    if i < len(task_result.handedness) and task_result.handedness[i]:
                        category = task_result.handedness[i][0]
                        hand_label = category.category_name or category.display_name
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand]
                    hands_metrics.append(self._build_hand_metrics(landmarks, hand_label))
                    draw_color = (95, 255, 115) if hand_label != "Right" else (120, 200, 255)
                    self._draw_landmarks_cv(frame, landmarks, hand_label=hand_label, line_color=draw_color)

        now = pytime.perf_counter()
        delta = max(now - self._last_ts, 1e-6)
        current = 1.0 / delta
        self.fps = current if self.fps == 0 else (self.fps * 0.9 + current * 0.1)
        self._last_ts = now

        return hands_metrics, frame, self.fps

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.hands is not None:
            self.hands.close()
        if self.hand_landmarker is not None:
            try:
                self.hand_landmarker.close()
            except RuntimeError as exc:
                # Interpreter shutdown can race with MediaPipe background dispatcher.
                if "cannot schedule new futures after shutdown" not in str(exc):
                    raise
            except Exception:
                pass
        self.cap.release()


class WebcamPanel(Entity):
    def __init__(self) -> None:
        super().__init__(
            parent=camera.ui,
            model="quad",
            color=color.rgba(12, 16, 30, 220),
            position=window.bottom_right + Vec2(-0.24, 0.16),
            scale=(0.44, 0.26),
        )

        self.panda_texture = PandaTexture("webcam_texture")
        self.panda_texture.setMagfilter(SamplerState.FT_linear)
        self.panda_texture.setMinfilter(SamplerState.FT_linear)
        self.texture_obj = Texture(self.panda_texture)
        self.texture_obj.filtering = "bilinear"
        self._size: tuple[int, int] | None = None

        self.image = Entity(
            parent=self,
            model="quad",
            texture=self.texture_obj,
            scale=(0.94, 0.9),
            z=-0.001,
            color=color.white,
        )

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Panda3D texture origin is bottom-left.
        rgb = np.flipud(rgb)
        height, width = rgb.shape[:2]

        if self._size != (width, height):
            self.panda_texture.setup2dTexture(width, height, PandaTexture.T_unsigned_byte, PandaTexture.F_rgb8)
            self._size = (width, height)
            self.panda_texture.setMagfilter(SamplerState.FT_linear)
            self.panda_texture.setMinfilter(SamplerState.FT_linear)

        self.panda_texture.setRamImageAs(rgb.tobytes(), "RGB")


class AstronautGestureController:
    def __init__(self, astronaut: Entity) -> None:
        self.astronaut = astronaut
        is_hubble = bool(getattr(astronaut, "is_hubble_model", False))
        self.is_hubble = is_hubble

        self.target_position = Vec3(0.0, 0.0, 0.0)
        self.base_scale = float(astronaut.scale_x)
        self.min_zoom_multiplier = 0.6 if is_hubble else 0.22
        self.max_zoom_multiplier = 15.84 if is_hubble else 7.92
        self.target_zoom_multiplier = 1.55 if is_hubble else 0.8
        self.single_rotation_y_gain = 520.0
        self.single_rotation_x_gain = 410.0
        self.single_rotation_z_gain = 240.0
        self.single_rotation_deadzone = 0.0035
        self.two_hand_move_x_gain = 8.4
        self.two_hand_move_y_gain = 5.4
        self.two_hand_zoom_power = 1.0
        self.two_hand_zoom_in_gain = 1.6
        self.two_hand_rotate_min_distance = 0.1
        self.two_hand_zoom_only_ratio = 0.58
        self.two_hand_pair_rotate_gain = 1.7
        self.two_hand_roll_rotate_gain = 2.2
        self.position_limit_x = 5.8 if is_hubble else 4.2
        self.position_limit_y = 3.4 if is_hubble else 3.0
        self.target_rotation_x = astronaut.rotation_x
        self.target_rotation_y = astronaut.rotation_y
        self.target_rotation_z = astronaut.rotation_z
        self.home_position = Vec3(
            float(astronaut.position.x),
            float(astronaut.position.y),
            float(astronaut.position.z),
        )
        self.home_rotation_x = float(astronaut.rotation_x)
        self.home_rotation_y = float(astronaut.rotation_y)
        self.home_rotation_z = float(astronaut.rotation_z)
        self.home_zoom_multiplier = self.target_zoom_multiplier

        self.prev_single_index_tip: Optional[tuple[float, float]] = None
        self.prev_single_roll_angle: Optional[float] = None
        self.two_hand_grab_active = False
        self.two_hand_anchor_center: Optional[tuple[float, float]] = None
        self.two_hand_anchor_position: Optional[Vec3] = None
        self.two_hand_anchor_distance: Optional[float] = None
        self.two_hand_anchor_zoom: Optional[float] = None
        self.two_hand_anchor_angle: Optional[float] = None
        self.two_hand_anchor_left_roll: Optional[float] = None
        self.two_hand_anchor_right_roll: Optional[float] = None
        self.two_hand_anchor_rotation_y: Optional[float] = None
        self.two_hand_anchor_rotation_z: Optional[float] = None

        self.gesture_state = "Idle"

    @staticmethod
    def _pick_primary_pair(hands: list[HandMetrics]) -> tuple[HandMetrics, HandMetrics]:
        left = next((h for h in hands if h.handedness == "Left"), None)
        right = next((h for h in hands if h.handedness == "Right"), None)
        if left is not None and right is not None:
            return left, right

        sorted_by_x = sorted(hands[:2], key=lambda h: h.palm_center[0])
        return sorted_by_x[0], sorted_by_x[1]

    @staticmethod
    def _single_hand_roll_angle(hand: HandMetrics) -> float:
        # Palm width direction (index MCP -> pinky MCP) gives stable in-plane hand roll.
        ix, iy, _ = hand.landmarks[5]
        px, py, _ = hand.landmarks[17]
        return float(np.arctan2(iy - py, ix - px))

    @staticmethod
    def _signed_angle_delta(current: float, anchor: float) -> float:
        return float(np.arctan2(np.sin(current - anchor), np.cos(current - anchor)))

    def _clamp_position(self, value: Vec3) -> Vec3:
        return Vec3(
            max(-self.position_limit_x, min(self.position_limit_x, float(value.x))),
            max(-self.position_limit_y, min(self.position_limit_y, float(value.y))),
            0.0,
        )

    def reset_view(self, hard: bool = False) -> None:
        self.target_position = Vec3(
            float(self.home_position.x),
            float(self.home_position.y),
            float(self.home_position.z),
        )
        self.target_rotation_x = self.home_rotation_x
        self.target_rotation_y = self.home_rotation_y
        self.target_rotation_z = self.home_rotation_z
        self.target_zoom_multiplier = self.home_zoom_multiplier
        self.on_hand_lost()
        if hard:
            self.astronaut.position = Vec3(
                float(self.home_position.x),
                float(self.home_position.y),
                float(self.home_position.z),
            )
            self.astronaut.rotation_x = self.home_rotation_x
            self.astronaut.rotation_y = self.home_rotation_y
            self.astronaut.rotation_z = self.home_rotation_z
            self.astronaut.scale = self.base_scale * self.home_zoom_multiplier

    def apply(self, hands: list[HandMetrics]) -> None:
        if not hands:
            self.on_hand_lost()
            return

        rot_delta = 0.0
        move_delta = 0.0
        zoom_delta = 0.0
        mode_suffix = "1H"
        active = []

        if len(hands) >= 2:
            left, right = self._pick_primary_pair(hands)
            mode_suffix = "2H"
            control_center = (
                (left.palm_center[0] + right.palm_center[0]) * 0.5,
                (left.palm_center[1] + right.palm_center[1]) * 0.5,
            )
            pair_angle = float(
                np.arctan2(
                    right.palm_center[1] - left.palm_center[1],
                    right.palm_center[0] - left.palm_center[0],
                )
            )
            left_roll = self._single_hand_roll_angle(left)
            right_roll = self._single_hand_roll_angle(right)
            hand_distance = float(
                np.hypot(
                    right.palm_center[0] - left.palm_center[0],
                    right.palm_center[1] - left.palm_center[1],
                )
            )
            hand_distance = max(hand_distance, 1e-6)

            if not self.two_hand_grab_active:
                self.two_hand_grab_active = True
                self.two_hand_anchor_center = control_center
                self.two_hand_anchor_position = Vec3(
                    float(self.target_position.x),
                    float(self.target_position.y),
                    float(self.target_position.z),
                )
                self.two_hand_anchor_distance = hand_distance
                self.two_hand_anchor_zoom = self.target_zoom_multiplier
                self.two_hand_anchor_angle = pair_angle
                self.two_hand_anchor_left_roll = left_roll
                self.two_hand_anchor_right_roll = right_roll
                self.two_hand_anchor_rotation_y = float(self.target_rotation_y)
                self.two_hand_anchor_rotation_z = float(self.target_rotation_z)

            if (
                self.two_hand_anchor_center is not None
                and self.two_hand_anchor_position is not None
                and self.two_hand_anchor_distance is not None
                and self.two_hand_anchor_zoom is not None
                and self.two_hand_anchor_angle is not None
                and self.two_hand_anchor_left_roll is not None
                and self.two_hand_anchor_right_roll is not None
                and self.two_hand_anchor_rotation_y is not None
                and self.two_hand_anchor_rotation_z is not None
            ):
                dx = control_center[0] - self.two_hand_anchor_center[0]
                dy = control_center[1] - self.two_hand_anchor_center[1]
                self.target_position = self._clamp_position(
                    self.two_hand_anchor_position
                    + Vec3(
                        dx * self.two_hand_move_x_gain,
                        -dy * self.two_hand_move_y_gain,
                        0.0,
                    )
                )

                # Requested behavior: hands closer -> zoom in, farther -> zoom out.
                dist_ratio = self.two_hand_anchor_distance / hand_distance
                if dist_ratio > 1.0:
                    dist_ratio = 1.0 + ((dist_ratio - 1.0) * self.two_hand_zoom_in_gain)
                zoom_target = self.two_hand_anchor_zoom * (dist_ratio ** self.two_hand_zoom_power)
                self.target_zoom_multiplier = max(
                    self.min_zoom_multiplier,
                    min(self.max_zoom_multiplier, zoom_target),
                )
                # When hands are very close, pair-angle becomes noisy.
                # Lock to zoom-only mode to prevent sudden spins.
                rotate_lock_distance = max(
                    self.two_hand_rotate_min_distance,
                    self.two_hand_anchor_distance * self.two_hand_zoom_only_ratio,
                )
                zoom_only_mode = hand_distance <= rotate_lock_distance

                if zoom_only_mode:
                    self.target_rotation_y = float(self.astronaut.rotation_y)
                    self.target_rotation_z = float(self.astronaut.rotation_z)
                    self.two_hand_anchor_angle = pair_angle
                    self.two_hand_anchor_left_roll = left_roll
                    self.two_hand_anchor_right_roll = right_roll
                    self.two_hand_anchor_rotation_y = float(self.target_rotation_y)
                    self.two_hand_anchor_rotation_z = float(self.target_rotation_z)
                    rot_delta = 0.0
                else:
                    pair_delta = self._signed_angle_delta(pair_angle, self.two_hand_anchor_angle)
                    left_roll_delta = self._signed_angle_delta(left_roll, self.two_hand_anchor_left_roll)
                    right_roll_delta = self._signed_angle_delta(right_roll, self.two_hand_anchor_right_roll)
                    roll_delta = (left_roll_delta + right_roll_delta) * 0.5

                    self.target_rotation_y = self.two_hand_anchor_rotation_y - (
                        np.degrees(roll_delta) * self.two_hand_roll_rotate_gain
                    )
                    self.target_rotation_z = self.two_hand_anchor_rotation_z - (
                        np.degrees(pair_delta) * self.two_hand_pair_rotate_gain
                    )
                    rot_delta = max(abs(pair_delta), abs(roll_delta))
                move_delta = float(np.hypot(dx, dy))
                zoom_delta = abs(hand_distance - self.two_hand_anchor_distance)

            self.prev_single_index_tip = None
            self.prev_single_roll_angle = None

            if rot_delta > 0.01:
                active.append("Rotation")
            if move_delta > 0.003:
                active.append("Translation")
            if zoom_delta > 0.004:
                active.append("Zoom")
        else:
            # 1-hand mode: rotation only.
            if self.two_hand_grab_active:
                self.two_hand_grab_active = False
                self.two_hand_anchor_center = None
                self.two_hand_anchor_position = None
                self.two_hand_anchor_distance = None
                self.two_hand_anchor_zoom = None
                self.two_hand_anchor_angle = None
                self.two_hand_anchor_left_roll = None
                self.two_hand_anchor_right_roll = None
                self.two_hand_anchor_rotation_y = None
                self.two_hand_anchor_rotation_z = None

            hand = hands[0]
            if self.prev_single_index_tip is not None:
                dx = hand.index_tip[0] - self.prev_single_index_tip[0]
                dy = hand.index_tip[1] - self.prev_single_index_tip[1]
                rot_delta = float(np.hypot(dx, dy))
                if rot_delta > self.single_rotation_deadzone:
                    self.target_rotation_y -= dx * self.single_rotation_y_gain
                    self.target_rotation_x += dy * self.single_rotation_x_gain
            self.prev_single_index_tip = hand.index_tip

            current_roll_angle = self._single_hand_roll_angle(hand)
            if self.prev_single_roll_angle is not None:
                d_roll = float(
                    np.arctan2(
                        np.sin(current_roll_angle - self.prev_single_roll_angle),
                        np.cos(current_roll_angle - self.prev_single_roll_angle),
                    )
                )
                self.target_rotation_z -= np.degrees(d_roll) * (self.single_rotation_z_gain / 180.0)
                rot_delta = max(rot_delta, abs(d_roll))
            self.prev_single_roll_angle = current_roll_angle

            if rot_delta > self.single_rotation_deadzone:
                active.append("Rotation")

        state = " + ".join(active) if active else "Tracking"
        self.gesture_state = f"{state} ({mode_suffix})"

    def on_hand_lost(self) -> None:
        self.prev_single_index_tip = None
        self.prev_single_roll_angle = None
        self.two_hand_grab_active = False
        self.two_hand_anchor_center = None
        self.two_hand_anchor_position = None
        self.two_hand_anchor_distance = None
        self.two_hand_anchor_zoom = None
        self.two_hand_anchor_angle = None
        self.two_hand_anchor_left_roll = None
        self.two_hand_anchor_right_roll = None
        self.two_hand_anchor_rotation_y = None
        self.two_hand_anchor_rotation_z = None
        self.gesture_state = "Hand Lost"

    def smooth_update(self) -> None:
        blend_transform = min(1.0, time.dt * 9.0)
        blend_scale = min(1.0, time.dt * 10.0)

        self.target_position = self._clamp_position(self.target_position)
        self.target_zoom_multiplier = max(
            self.min_zoom_multiplier,
            min(self.max_zoom_multiplier, self.target_zoom_multiplier),
        )
        self.astronaut.position = lerp(self.astronaut.position, self.target_position, blend_transform)
        self.astronaut.rotation_x = lerp(self.astronaut.rotation_x, self.target_rotation_x, blend_transform)
        self.astronaut.rotation_y = lerp(self.astronaut.rotation_y, self.target_rotation_y, blend_transform)
        self.astronaut.rotation_z = lerp(self.astronaut.rotation_z, self.target_rotation_z, blend_transform)

        current_multiplier = max(self.astronaut.scale_x / max(self.base_scale, 1e-6), 1e-6)
        zoom_multiplier = lerp(current_multiplier, self.target_zoom_multiplier, blend_scale)
        self.astronaut.scale = self.base_scale * zoom_multiplier


class SceneRuntime(Entity):
    def __init__(self) -> None:
        super().__init__()
        try:
            self.setShaderOff(100)
        except Exception:
            pass

        window.title = "Astro Gesture 3D"
        window.color = color.black
        window.fps_counter.enabled = False
        try:
            # Some macOS/OpenGL environments fail generated GLSL shaders.
            # Force fixed pipeline for stable visibility.
            # Do not apply this to UI/render2d; UI uses transparent quads.
            for node_name in ("render", "scene"):
                node = getattr(builtins, node_name, None)
                if node is not None and hasattr(node, "setShaderOff"):
                    node.setShaderOff(100)
        except Exception:
            pass

        try:
            if hasattr(render, "clearFog"):
                render.clearFog()
            if hasattr(scene, "clearFog"):
                scene.clearFog()
        except Exception:
            pass

        self._build_environment()
        self.astronaut = self._load_astronaut()
        self.controller = AstronautGestureController(self.astronaut)
        self.controller.reset_view(hard=True)
        self.webcam_panel = WebcamPanel() if SHOW_WEBCAM_PANEL else None

        top_left = window.top_left + Vec2(0.03, -0.04)
        self.fps_text = Text(parent=camera.ui, text="FPS: --", position=top_left, origin=(-0.5, 0.5), scale=1.1)
        self.gesture_text = Text(
            parent=camera.ui,
            text="Gesture: Idle",
            position=top_left + Vec2(0.0, -0.06),
            origin=(-0.5, 0.5),
            scale=1.1,
        )
        self.hand_status_text = Text(
            parent=camera.ui,
            text="Hand Lost",
            position=top_left + Vec2(0.0, -0.12),
            origin=(-0.5, 0.5),
            scale=1.1,
            color=color.red,
            enabled=False,
        )

        self.hand_tracker: Optional[HandTracker] = None
        self.tracker_error: Optional[str] = None
        self.hand_lost_time = 0.0
        self._startup_shader_guard_frames = 12
        try:
            self.hand_tracker = HandTracker()
            atexit.register(self.hand_tracker.close)
        except RuntimeError as exc:
            self.tracker_error = str(exc)
            self.hand_status_text.text = self.tracker_error
            self.hand_status_text.enabled = True

        self._force_shader_off_everything()

    @staticmethod
    def _force_shader_off_tree(root: object) -> None:
        try:
            if hasattr(root, "setShaderOff"):
                root.setShaderOff(100)
        except Exception:
            pass

        children = getattr(root, "children", None)
        if children:
            for child in list(children):
                SceneRuntime._force_shader_off_tree(child)

        model = getattr(root, "model", None)
        if model is not None:
            try:
                if hasattr(model, "setShaderOff"):
                    model.setShaderOff(100)
                if hasattr(model, "findAllMatches"):
                    for node in model.findAllMatches("**"):
                        if hasattr(node, "setShaderOff"):
                            node.setShaderOff(100)
            except Exception:
                pass

    def _force_shader_off_everything(self) -> None:
        for node_name in ("render", "scene"):
            node = getattr(builtins, node_name, None)
            if node is not None:
                self._force_shader_off_tree(node)

    def _build_environment(self) -> None:
        try:
            if hasattr(render, "clearFog"):
                render.clearFog()
            if hasattr(scene, "clearFog"):
                scene.clearFog()
        except Exception:
            pass

        camera.position = (0, 0, -10.5)
        camera.fov = 58
        camera.clip_plane_near = 0.01
        camera.clip_plane_far = 500.0

        starfield = Entity(parent=scene, name="starfield")
        try:
            starfield.setShaderOff(100)
            starfield.setLightOff(1)
            starfield.setMaterialOff(1)
        except Exception:
            pass
        for _ in range(260):
            direction = Vec3(
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
            ).normalized()
            radius = random.uniform(30.0, 52.0)
            star = Entity(
                parent=starfield,
                model="sphere",
                scale=random.uniform(0.04, 0.1),
                position=direction * radius,
                color=color.rgba(245, 248, 255, random.randint(170, 255)),
            )
            try:
                star.setLightOff(1)
                star.setMaterialOff(1)
                star.setShaderOff(100)
            except Exception:
                pass

    def _load_astronaut(self) -> Entity:
        asset_root = Path(__file__).resolve().parent
        self._ensure_decoded_glb(asset_root, source_name="Astronaut.glb", decoded_name="Astronaut_plain.glb")
        self._ensure_decoded_glb(
            asset_root,
            source_name="Hubble Space Telescope (B).glb",
            decoded_name="Hubble Space Telescope (B)_plain.glb",
        )
        self._ensure_compatible_glb(asset_root)

        hubble_present = (asset_root / "Hubble Space Telescope (B).glb").exists()
        if hubble_present:
            # If user places Hubble model, do not silently fall back to astronaut.
            candidates = [
                "Hubble Space Telescope (B)_plain.glb",
                "Hubble Space Telescope (B).glb",
            ]
        else:
            candidates = [
                "Astronaut_plain.obj",
                "Astronaut_compat.glb",
                "Astronaut_plain.glb",
                "Astronaut.glb",
                "Astronaut_converted.obj",
            ]

        model_data = None
        loaded_name = None
        for name in candidates:
            path = asset_root / name
            if not path.exists():
                continue
            try:
                if name.lower().endswith(".obj"):
                    model_data = builtins.loader.loadModel(str(path))
                    if hasattr(model_data, "isEmpty") and model_data.isEmpty():
                        model_data = None
                else:
                    try:
                        model_data = load_model(name)
                    except Exception:
                        model_data = None
                    if model_data is None or (hasattr(model_data, "isEmpty") and model_data.isEmpty()):
                        model_data = builtins.loader.loadModel(str(path))
                    if hasattr(model_data, "isEmpty") and model_data.isEmpty():
                        model_data = None
            except Exception as exc:
                print(f"warning: failed to load {name}: {exc}")
                model_data = None
            if model_data is not None:
                loaded_name = name
                break

        if model_data is None and hubble_present:
            raise FileNotFoundError(
                "Hubble Space Telescope model exists but could not be loaded. "
                "Try installing Node.js (for npx) so a decoded copy can be generated, "
                "or provide Hubble Space Telescope (B)_plain.glb."
            )

        if model_data is None:
            raise FileNotFoundError(
                "No compatible model found. Expected Astronaut_plain.obj, Astronaut_compat.glb, Astronaut_plain.glb, Astronaut.glb, or Astronaut_converted.obj."
            )

        astronaut = Entity(
            model=model_data,
            scale=1.0,
            position=(0, -0.2, 0),
            rotation=(0, 180, 0),
        )
        try:
            astronaut.setShaderOff(100)
            astronaut.clearFog()
        except Exception:
            pass
        is_hubble = loaded_name is not None and loaded_name.startswith("Hubble Space Telescope (B)")
        self._normalize_model_transform(astronaut, target_max_extent=5.2 if is_hubble else 2.2)
        astronaut.is_hubble_model = is_hubble
        if is_hubble:
            astronaut.position = (0, 0, 0)
            astronaut.rotation = (0, 132, 0)
            self._apply_hubble_visuals(astronaut)
        else:
            self._apply_compat_shader(astronaut)
        print(f"info: model loaded: {loaded_name}")
        return astronaut

    @staticmethod
    def _ensure_decoded_glb(asset_root: Path, source_name: str, decoded_name: str) -> None:
        src = asset_root / source_name
        decoded = asset_root / decoded_name
        if not src.exists() or decoded.exists():
            return

        npx = shutil.which("npx")
        if npx is None:
            return

        try:
            subprocess.run(
                [npx, "-y", "@gltf-transform/cli", "copy", str(src), str(decoded)],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"info: created {decoded_name} from {source_name}")
        except Exception as exc:
            print(f"warning: failed to create {decoded_name}: {exc}")

    @staticmethod
    def _ensure_compatible_glb(asset_root: Path) -> None:
        src = asset_root / "Astronaut_plain.glb"
        dst = asset_root / "Astronaut_compat.glb"
        if not src.exists() or dst.exists():
            return

        try:
            from pygltflib import GLTF2  # Optional dependency for texture compatibility patch.
        except Exception:
            return

        try:
            gltf = GLTF2().load(str(src))
            if not gltf.textures or not gltf.images:
                gltf.save(str(dst))
                return

            changed = False
            for tex in gltf.textures:
                ext = tex.extensions or {}
                webp_ext = ext.get("EXT_texture_webp") if isinstance(ext, dict) else None
                if not webp_ext:
                    continue

                webp_idx = webp_ext.get("source")
                if webp_idx is None:
                    continue

                webp_img = gltf.images[webp_idx]
                png_idx = None
                for i, img in enumerate(gltf.images):
                    if img.name == webp_img.name and img.mimeType == "image/png":
                        png_idx = i
                        break

                if png_idx is None:
                    continue

                tex.source = png_idx
                tex.extensions = None
                changed = True

            if changed:
                if gltf.extensionsUsed:
                    gltf.extensionsUsed = [e for e in gltf.extensionsUsed if e != "EXT_texture_webp"]
                if gltf.extensionsRequired:
                    gltf.extensionsRequired = [e for e in gltf.extensionsRequired if e != "EXT_texture_webp"]

            gltf.save(str(dst))
            print("info: created Astronaut_compat.glb with PNG texture fallback")
        except Exception as exc:
            print(f"warning: failed to create Astronaut_compat.glb: {exc}")

    @staticmethod
    def _normalize_model_transform(entity: Entity, target_max_extent: float = 2.2) -> None:
        model = entity.model
        if model is None:
            return

        center: Vec3 | None = None
        extent: Vec3 | None = None

        if hasattr(model, "getTightBounds"):
            try:
                bounds = model.getTightBounds()
            except Exception:
                bounds = None

            if bounds and len(bounds) == 2 and bounds[0] is not None and bounds[1] is not None:
                min_v, max_v = bounds
                center = Vec3(
                    float((min_v.x + max_v.x) / 2.0),
                    float((min_v.y + max_v.y) / 2.0),
                    float((min_v.z + max_v.z) / 2.0),
                )
                extent = Vec3(
                    float(max_v.x - min_v.x),
                    float(max_v.y - min_v.y),
                    float(max_v.z - min_v.z),
                )

        if extent is None and hasattr(model, "vertices"):
            vertices = getattr(model, "vertices", None)
            if vertices:
                xs = [float(v[0]) for v in vertices]
                ys = [float(v[1]) for v in vertices]
                zs = [float(v[2]) for v in vertices]
                center = Vec3(
                    (min(xs) + max(xs)) / 2.0,
                    (min(ys) + max(ys)) / 2.0,
                    (min(zs) + max(zs)) / 2.0,
                )
                extent = Vec3(
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                    max(zs) - min(zs),
                )

        if center is None or extent is None:
            return

        max_extent = max(extent.x, extent.y, extent.z, 1e-6)
        scale_factor = target_max_extent / max_extent

        if hasattr(model, "setPos"):
            try:
                model.setPos(-center.x, -center.y, -center.z)
            except Exception:
                pass
        entity.scale = scale_factor

    @staticmethod
    def _apply_compat_shader(entity: Entity) -> None:
        model = entity.model
        if model is None or not hasattr(model, "setShader"):
            return
        try:
            model.setShader(get_legacy_texture_shader())
            # Converted OBJ/GLB assets can have mixed winding order.
            # Render both sides to avoid visible holes from back-face culling.
            model.setTwoSided(True)
            for node in model.findAllMatches("**"):
                if hasattr(node, "setTwoSided"):
                    node.setTwoSided(True)
        except Exception as exc:
            print(f"warning: failed to apply compatibility shader: {exc}")

    @staticmethod
    def _apply_hubble_visuals(entity: Entity) -> None:
        model = entity.model
        if model is None:
            return
        try:
            if hasattr(entity, "setShaderOff"):
                entity.setShaderOff(100)
            if hasattr(entity, "setLightOff"):
                entity.setLightOff(1)
            if hasattr(entity, "setMaterialOff"):
                entity.setMaterialOff(1)
            if hasattr(entity, "setTextureOff"):
                entity.setTextureOff(1)
            if hasattr(entity, "setTransparency"):
                entity.setTransparency(TransparencyAttrib.M_alpha)
            if hasattr(entity, "setAlphaScale"):
                entity.setAlphaScale(HUBBLE_VISUAL_ALPHA)
            if hasattr(entity, "clearFog"):
                entity.clearFog()
            # Avoid custom GLSL here; some macOS drivers in this setup reject generated shader versions.
            # Use fixed pipeline state so geometry stays visible.
            if hasattr(model, "clearShader"):
                model.clearShader()
            if hasattr(model, "clearFog"):
                model.clearFog()
            if hasattr(model, "setShaderOff"):
                model.setShaderOff(1)
            if hasattr(model, "setLightOff"):
                model.setLightOff(1)
            if hasattr(model, "setMaterialOff"):
                model.setMaterialOff(1)
            if hasattr(model, "setTextureOff"):
                model.setTextureOff(1)
            if hasattr(model, "setColor"):
                model.setColor(
                    HUBBLE_VISUAL_TINT[0],
                    HUBBLE_VISUAL_TINT[1],
                    HUBBLE_VISUAL_TINT[2],
                    HUBBLE_VISUAL_ALPHA,
                )
            if hasattr(model, "setColorScale"):
                model.setColorScale(1.0, 1.0, 1.0, HUBBLE_VISUAL_ALPHA)
            if hasattr(model, "setTransparency"):
                model.setTransparency(TransparencyAttrib.M_alpha)
            if hasattr(model, "setAlphaScale"):
                model.setAlphaScale(HUBBLE_VISUAL_ALPHA)
            if hasattr(model, "setTwoSided"):
                model.setTwoSided(True)
            if hasattr(model, "setAttrib"):
                model.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
            for node in model.findAllMatches("**"):
                if hasattr(node, "clearShader"):
                    node.clearShader()
                if hasattr(node, "setShaderOff"):
                    node.setShaderOff(1)
                if hasattr(node, "setLightOff"):
                    node.setLightOff(1)
                if hasattr(node, "setMaterialOff"):
                    node.setMaterialOff(1)
                if hasattr(node, "setTextureOff"):
                    node.setTextureOff(1)
                if hasattr(node, "setColor"):
                    node.setColor(
                        HUBBLE_VISUAL_TINT[0],
                        HUBBLE_VISUAL_TINT[1],
                        HUBBLE_VISUAL_TINT[2],
                        HUBBLE_VISUAL_ALPHA,
                    )
                if hasattr(node, "setColorScale"):
                    node.setColorScale(1.0, 1.0, 1.0, HUBBLE_VISUAL_ALPHA)
                if hasattr(node, "setTransparency"):
                    node.setTransparency(TransparencyAttrib.M_alpha)
                if hasattr(node, "setAlphaScale"):
                    node.setAlphaScale(HUBBLE_VISUAL_ALPHA)
                if hasattr(node, "clearFog"):
                    node.clearFog()
                if hasattr(node, "setTwoSided"):
                    node.setTwoSided(True)
                if hasattr(node, "setAttrib"):
                    node.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
            SceneRuntime._attach_hubble_glow(entity)
        except Exception as exc:
            print(f"warning: failed to apply hubble visuals: {exc}")

    @staticmethod
    def _attach_hubble_glow(entity: Entity) -> None:
        model = entity.model
        if model is None or not hasattr(model, "copyTo"):
            return

        prev_glow = getattr(entity, "_hubble_glow_shell", None)
        if prev_glow is not None and hasattr(prev_glow, "removeNode"):
            try:
                prev_glow.removeNode()
            except Exception:
                pass

        glow = model.copyTo(entity)
        entity._hubble_glow_shell = glow
        if hasattr(glow, "setScale"):
            glow.setScale(HUBBLE_GLOW_SCALE)

        try:
            glow.clearFog()
        except Exception:
            pass

        targets = [glow]
        try:
            targets.extend(list(glow.findAllMatches("**")))
        except Exception:
            pass

        for node in targets:
            try:
                if hasattr(node, "clearShader"):
                    node.clearShader()
                if hasattr(node, "setShaderOff"):
                    node.setShaderOff(1)
                if hasattr(node, "setLightOff"):
                    node.setLightOff(1)
                if hasattr(node, "setMaterialOff"):
                    node.setMaterialOff(1)
                if hasattr(node, "setTextureOff"):
                    node.setTextureOff(1)
                if hasattr(node, "setTransparency"):
                    blend_mode = getattr(TransparencyAttrib, "M_add", TransparencyAttrib.M_alpha)
                    node.setTransparency(blend_mode)
                if hasattr(node, "setColor"):
                    node.setColor(
                        HUBBLE_GLOW_COLOR[0],
                        HUBBLE_GLOW_COLOR[1],
                        HUBBLE_GLOW_COLOR[2],
                        HUBBLE_GLOW_ALPHA,
                    )
                if hasattr(node, "setColorScale"):
                    node.setColorScale(1.25, 1.35, 1.6, HUBBLE_GLOW_ALPHA)
                if hasattr(node, "setAlphaScale"):
                    node.setAlphaScale(HUBBLE_GLOW_ALPHA)
                if hasattr(node, "setTwoSided"):
                    node.setTwoSided(True)
                if hasattr(node, "setAttrib"):
                    node.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
                if hasattr(node, "setDepthWrite"):
                    node.setDepthWrite(False)
                if hasattr(node, "setDepthTest"):
                    node.setDepthTest(False)
                if hasattr(node, "setBin"):
                    node.setBin("transparent", 48)
                if hasattr(node, "clearFog"):
                    node.clearFog()
            except Exception:
                pass

    def update(self) -> None:
        if self._startup_shader_guard_frames > 0:
            self._force_shader_off_everything()
            try:
                if hasattr(render, "clearFog"):
                    render.clearFog()
                if hasattr(scene, "clearFog"):
                    scene.clearFog()
            except Exception:
                pass
            self._startup_shader_guard_frames -= 1

        if self.hand_tracker is None:
            return

        hands, frame, fps = self.hand_tracker.read()
        if self.webcam_panel is not None:
            self.webcam_panel.update_frame(frame)

        if not hands:
            self.controller.on_hand_lost()
            self.hand_status_text.enabled = True
            self.hand_lost_time += max(time.dt, 0.0)
            if self.hand_lost_time > 1.25:
                self.controller.reset_view(hard=False)
        else:
            self.controller.apply(hands)
            self.hand_status_text.enabled = False
            self.hand_lost_time = 0.0

        self.controller.smooth_update()
        self.fps_text.text = f"FPS: {fps:5.1f}"
        self.gesture_text.text = f"Gesture: {self.controller.gesture_state}"

    def input(self, key: str) -> None:
        if key == "r":
            self.controller.reset_view(hard=True)
            return
        if key == "escape":
            if self.hand_tracker is not None:
                self.hand_tracker.close()
                self.hand_tracker = None
            application.quit()


if __name__ == "__main__":
    app = Ursina(borderless=False)
    SceneRuntime()
    app.run()
