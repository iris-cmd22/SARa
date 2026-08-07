from dataclasses import dataclass


@dataclass
class Plan:
    algorithm: str
    budget_m: float
    num_drones: int = 1
    altitude_m: float = 80.0
    fov_deg: float = 45.0
    overlap_ratio: float = 0.0
    status: str = "SEARCHING"  # SEARCHING | RETURN_TO_BASE
