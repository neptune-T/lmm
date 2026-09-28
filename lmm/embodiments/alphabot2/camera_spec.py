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

Live-verified on 2026-09-28 (Isaac Sim 4.5, RTX, see
``docs/ALPHABOT2_ISAAC_LIVE_SMOKE.md``): 240x320x3 uint8 RGB, channel order
confirmed with red/green/blue markers, head view upright and not mirrored, all
three cameras rendered on the same frame. The asset ships camera *mounts* but
no camera prims, so the extrinsics below are spec/geometry-derived, not the
vendor's calibration.
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

#: USD prims the live backend attaches camera prims to, and how.
USD_CAMERA_MOUNTS: dict[str, dict[str, object]] = {
    "cam_front": {
        "parent": "head_link2/head_camera",
        "convention": "optical frame: +z optical axis, +x image right, +y image down",
        "measured_pitch_down_deg": 29.96,
        "xacro_cam_high_pitch_down_deg": 37.0,   # rpy 0 0.6458 0 -- the two public descriptions differ by 7 deg
        "hfov_deg": 69.0,                         # RealSense D435 colour
        "near_clip_m": 0.10,                      # clears the head shell around the mount
    },
    "cam_left_wrist": {
        "parent": "Arm_Left/link7",
        "position": "the D405 prim of the ee_fb_405 bracket",
        "optical_axis": "towards pad midpoint + 0.10 m along the approach",
        "image_up": "away from the gripper axis (fingers at the image bottom)",
        "hfov_deg": 87.0,                         # RealSense D405
        "near_clip_m": 0.02,
    },
    "cam_right_wrist": {"mirror_of": "cam_left_wrist", "parent": "Arm_Right/link7"},
}
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
SIM_RESOLUTION = ALPHABOT2_ACCOUNT_RESOLUTION     # rendered live at 240x320


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
            "note": "Flags are True only for LIVE_QUALIFIED, measured in Isaac "
                    "on the public Ai2 Bot2 asset; extrinsics are "
                    "spec/geometry-derived, not vendor-calibrated.",
        }


#: The configuration qualified live on 2026-09-28.
LIVE_QUALIFIED = CameraContract(resolution=SIM_RESOLUTION, frames_verified=True,
                                orientation_verified=True, channel_order_verified=True)
