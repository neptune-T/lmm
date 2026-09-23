"""AlphaBot 2 v1 camera contract.

The robot exposes head, chest and back RGB (1920x1080) plus both wrists
(1280x720 RGB-D). pi0.5 consumes a base view plus two wrists, so v1 uses the
smallest set that maps cleanly onto that.

**Head is chosen over chest as the base view.** The head sits on a 2-DoF neck
at the top of a 180 cm robot, giving the widest forward view and the best
sight-line over receptacles during navigation. Chest is recorded as the
fallback: it is rigidly mounted, so it is the *more stable* view if the neck
turns out to move during manipulation. That trade-off cannot be settled
without the robot, so both are named and the choice is revisitable.

Depth is excluded: pi0.5 is RGB, and depth has no counterpart in the RoboCOIN
convention.
"""

from __future__ import annotations

from dataclasses import dataclass

CANONICAL_CAMERAS: tuple[str, ...] = ("cam_front", "cam_left_wrist", "cam_right_wrist")

MODEL_KEYS: dict[str, str] = {
    "cam_front": "base_0_rgb",
    "cam_left_wrist": "left_wrist_0_rgb",
    "cam_right_wrist": "right_wrist_0_rgb",
}

#: Public dataset name -> canonical key. Resolutions are the recorded ones,
#: not the sensor natives.
ROBOCOIN_MAPPING = {
    "observation.images.cam_front_head_rgb": "cam_front",
    "observation.images.cam_left_wrist_rgb": "cam_left_wrist",
    "observation.images.cam_right_wrist_rgb": "cam_right_wrist",
}
ROBOCOIN_RESOLUTION = (480, 640)
ROBOCOIN_ALTERNATE_BASE = "observation.images.cam_front_chest_rgb"

#: UNVERIFIED: this source's names are positional and it is not established
#: that camera_left/right are wrist cameras rather than third-person views.
ALPHABOT2_ACCOUNT_MAPPING_UNVERIFIED = {
    "observation.images.camera_front": "cam_front",
    "observation.images.camera_left": "cam_left_wrist",
    "observation.images.camera_right": "cam_right_wrist",
}
ALPHABOT2_ACCOUNT_RESOLUTION = (240, 320)

EXCLUDED = {
    "back_rgb": "no counterpart in public data; not used for forward manipulation",
    "chest_rgb": "recorded as fallback base view, not a fourth input",
    "wrist_depth": "pi0.5 is RGB-only; absent from the RoboCOIN convention",
    "camera_top": "alphabot2-account only; no vendor or RoboCOIN counterpart",
}


@dataclass(frozen=True)
class CameraContract:
    """Resolved camera configuration."""

    base_view: str = "head"
    resolution: tuple[int, int] | None = None
    frame_names_verified: bool = False

    def describe(self) -> dict[str, object]:
        return {
            "canonical_cameras": list(CANONICAL_CAMERAS),
            "model_keys": dict(MODEL_KEYS),
            "base_view": self.base_view,
            "alternate_base_view": "chest",
            "depth": False,
            "resolution": self.resolution,
            "frame_names_verified": self.frame_names_verified,
            "note": "Exact simulator/SDK frame names cannot be recorded until "
                    "a robot description or SDK is available.",
        }
