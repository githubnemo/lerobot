"""Configuration for reusable Video VAM dataset contracts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoVAMDatasetConfig:
    """Describe the data contract consumed by a video-action model."""

    repo_id: str
    revision: str
    codebase_version: str = "v3.0"
    fps: int = 10
    camera_key: str = "observation.images.front"
    camera_shape: tuple[int, int, int] = (480, 640, 3)
    state_key: str = "observation.state"
    action_key: str = "action"
    state_shape: tuple[int] = (6,)
    state_offsets: tuple[int, ...] = (0,)
    action_shape: tuple[int] = (6,)
    state_names: tuple[str, ...] = (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )
    action_names: tuple[str, ...] = (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )
    task: str = "take cube out of box"
    history_offsets: tuple[int, ...] = (-4, -3, -2, -1, 0)
    action_chunk_size: int = 30
    total_episodes: int = 40
    total_frames: int = 6536

    def __post_init__(self) -> None:
        """Reject an internally inconsistent contract configuration."""
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.history_offsets != (-4, -3, -2, -1, 0):
            raise ValueError(f"history_offsets must be (-4, -3, -2, -1, 0), got {self.history_offsets}")
        if self.state_offsets != (0,):
            raise ValueError(f"state_offsets must be (0,), got {self.state_offsets}")
        if len(self.state_names) != self.state_shape[0]:
            raise ValueError("state_names length must match state_shape")
        if len(self.action_names) != self.action_shape[0]:
            raise ValueError("action_names length must match action_shape")
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")

    @property
    def history_length(self) -> int:
        """Return the number of causal observations in one sample."""
        return len(self.history_offsets)

    @property
    def sample_camera_shape(self) -> tuple[int, int, int, int]:
        """Return the current LeRobot decoded camera layout ``TCHW``."""
        height, width, channels = self.camera_shape
        return (self.history_length, channels, height, width)

    @property
    def state_history_length(self) -> int:
        """Return the number of proprioception tokens in one sample."""
        return len(self.state_offsets)

    @property
    def sample_state_shape(self) -> tuple[int, int]:
        """Return the expanded state history shape."""
        return (self.state_history_length, self.state_shape[0])

    @property
    def sample_action_shape(self) -> tuple[int, int]:
        """Return the expanded action chunk shape."""
        return (self.action_chunk_size, self.action_shape[0])

    def delta_timestamps(self) -> dict[str, list[float]]:
        """Return LeRobot ``delta_timestamps`` for this contract."""
        return {
            self.camera_key: [offset / self.fps for offset in self.history_offsets],
            self.state_key: [offset / self.fps for offset in self.state_offsets],
            self.action_key: [offset / self.fps for offset in range(self.action_chunk_size)],
        }


CUBE_OUT_OF_BOX_CONTRACT = VideoVAMDatasetConfig(
    repo_id="hubnemo/cube_out_of_box_dataset",
    revision="243370c3c08bcbd860133c4a0d658ea7c1d2e77e",
)
