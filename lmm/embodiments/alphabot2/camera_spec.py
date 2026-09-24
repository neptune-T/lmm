"""AlphaBot 2 v1 camera contract, resolved against the Ai2 Bot2 asset.

Canonical visual input is the smallest set pi0.5 consumes naturally: one
forward view plus both wrists. Depth is not used in v1.

Asset camera frames (from the pinned Ai2 Bot2 description)::

    head_camera     fixed to head_link2   -- on the 2-DoF neck
    front_camera    fixed to body_link4   -- chest, rigid w.r.t. the torso
    back_camera     fixed to body_link4
    chassis_camera  fixed to base_link    -- Orbbec Dabai DW
    {side}_link8    RealSense D405        -- wrist cameras, one per arm

Forward-view choice: **head_camera**. The neck (``head_joint1``, ``head_joint2``)
is held fixed in v1, so the head view is rigid in practice while sitting higher
than the chest, which gives a better sight-line over receptacles during
navigation. ``front_camera`` (chest) is the documented fallback and becomes the
better choice the moment the neck is ever actuated.

Nothing here is verified against rendered frames -- resolution, orientation,
channel order and mirroring all require a running simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

CANONICAL_CAMERAS: tuple[str, ...] = ("cam_front", "cam_left_wrist", "cam_right_wrist")

MODEL_KEYS: dict[str, str] = {
    "cam_front": "base_0_rgb",
    "cam_left_wrist": "left_wrist_0_rgb",
    "cam_right_wrist": "right_wrist_0_rgb",
}

#: canonical key -> Ai2 Bot2 asset frame
ASSET_FRAMES: dict[str, str] = {
    "cam_front": "head_camera",
    "cam_left_wrist": "left_link8",
    "cam_right_wrist": "right_link8",
}

ALTERNATE_FRONT_FRAME = "front_camera"        # chest, rigid on body_link4
WRIST_SENSOR_MODEL = "RealSense D405"

EXCLUDED = {
    "back_camera": "rear-facing; no role in forward manipulation",
    "chassis_camera": "Orbbec Dabai DW on base_link; no counterpart in public data",
    "wrist_depth": "pi0.5 is RGB-only; depth deferred",
}

#: Public dataset mappings, for later data reuse.
ROBOCOIN_MAPPING = {
    "observation.images.cam_front_head_rgb": "cam_front",
    "observation.images.cam_left_wrist_rgb": "cam_left_wrist",
    "observation.images.cam_right_wrist_rgb": "cam_right_wrist",
}
ROBOCOIN_RESOLUTION = (480, 640)

#: UNVERIFIED: this source's names are positional and it is not established
#: that camera_left/right are wrist views rather than third-person views.
ALPHABOT2_ACCOUNT_MAPPING_UNVERIFIED = {
    "observation.images.camera_front": "cam_front",
    "observation.images.camera_left": "cam_left_wrist",
    "observation.images.camera_right": "cam_right_wrist",
}
ALPHABOT2_ACCOUNT_RESOLUTION = (240, 320)


@dataclass(frozen=True)
class CameraContract:
    """Resolved camera configuration."""

    front_frame: str = ASSET_FRAMES["cam_front"]
    resolution: tuple[int, int] | None = None
    frames_verified: bool = False
    orientation_verified: bool = False
    channel_order_verified: bool = False

    def describe(self) -> dict[str, object]:
        return {
            "canonical_cameras": list(CANONICAL_CAMERAS),
            "model_keys": dict(MODEL_KEYS),
            "asset_frames": dict(ASSET_FRAMES),
            "front_frame": self.front_frame,
            "alternate_front_frame": ALTERNATE_FRONT_FRAME,
            "wrist_sensor": WRIST_SENSOR_MODEL,
            "depth": False,
            "resolution": self.resolution,
            "frames_verified": self.frames_verified,
            "orientation_verified": self.orientation_verified,
            "channel_order_verified": self.channel_order_verified,
            "note": "Resolution, orientation, RGB channel order and mirroring "
                    "require a live render to verify.",
        }
