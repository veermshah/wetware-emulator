"""Virtual multi-electrode array geometry and electrode properties."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class MEA:
    """A planar grid of recording/stimulating electrodes.

    Electrodes are numbered row-major from the top-left corner, so electrode
    ``k`` sits at row ``k // cols`` and column ``k % cols``. Coordinates are
    in micrometres with the array centred on the origin.

    Args:
        rows: Number of electrode rows.
        cols: Number of electrode columns.
        pitch_um: Centre-to-centre spacing between electrodes.
        electrode_diameter_um: Electrode diameter, used for charge density.
        noise_rms_uv: Mean thermal noise of the recording channels (µV RMS).
        sample_rate_hz: Sampling rate of the synthesized voltage traces.
    """

    rows: int = 8
    cols: int = 8
    pitch_um: float = 200.0
    electrode_diameter_um: float = 30.0
    noise_rms_uv: float = 5.0
    sample_rate_hz: float = 10_000.0
    _dead: set[int] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError("MEA needs at least one row and one column")
        if self.pitch_um <= 0:
            raise ValueError("pitch_um must be positive")
        if self.sample_rate_hz < 2_000:
            raise ValueError("sample_rate_hz must be at least 2 kHz to resolve spikes")

    @property
    def n_electrodes(self) -> int:
        """Total number of electrodes."""
        return self.rows * self.cols

    @property
    def positions(self) -> NDArray[np.float64]:
        """``(n_electrodes, 2)`` array of electrode ``(x, y)`` centres in µm."""
        r, c = np.divmod(np.arange(self.n_electrodes), self.cols)
        x = (c - (self.cols - 1) / 2.0) * self.pitch_um
        y = (r - (self.rows - 1) / 2.0) * self.pitch_um
        return np.column_stack([x, y]).astype(float)

    @property
    def extent_um(self) -> tuple[float, float]:
        """Width and height spanned by electrode centres."""
        return ((self.cols - 1) * self.pitch_um, (self.rows - 1) * self.pitch_um)

    @property
    def electrode_area_cm2(self) -> float:
        """Geometric electrode area in cm²."""
        radius_cm = self.electrode_diameter_um * 1e-4 / 2.0
        return float(np.pi * radius_cm**2)

    @property
    def dead_electrodes(self) -> list[int]:
        """Sorted list of electrodes that no longer record or stimulate."""
        return sorted(self._dead)

    @property
    def alive_mask(self) -> NDArray[np.bool_]:
        """Boolean mask of working electrodes."""
        mask = np.ones(self.n_electrodes, dtype=bool)
        if self._dead:
            mask[list(self._dead)] = False
        return mask

    def mark_dead(self, electrodes: int | list[int]) -> None:
        """Mark electrodes as failed."""
        for e in np.atleast_1d(electrodes):
            self._check_index(int(e))
            self._dead.add(int(e))

    def index(self, row: int, col: int) -> int:
        """Electrode index for a ``(row, col)`` grid position."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(f"({row}, {col}) is outside a {self.rows}x{self.cols} array")
        return row * self.cols + col

    def grid_position(self, electrode: int) -> tuple[int, int]:
        """``(row, col)`` of an electrode index."""
        self._check_index(electrode)
        return divmod(electrode, self.cols)

    def as_grid(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reshape a per-electrode vector to a ``(rows, cols)`` grid."""
        return np.asarray(values).reshape(self.rows, self.cols)

    def _check_index(self, electrode: int) -> None:
        if not 0 <= electrode < self.n_electrodes:
            raise IndexError(f"electrode {electrode} out of range 0..{self.n_electrodes - 1}")

    def to_dict(self) -> dict[str, float | int | list[int]]:
        """JSON-friendly description."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "pitch_um": self.pitch_um,
            "electrode_diameter_um": self.electrode_diameter_um,
            "noise_rms_uv": self.noise_rms_uv,
            "sample_rate_hz": self.sample_rate_hz,
            "dead_electrodes": self.dead_electrodes,
        }
