from dataclasses import dataclass

import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import numpy as np
from mujoco.viewer import Handle as ViewerHandle


@dataclass
class State:
    mj_data: mujoco.MjData = None
    mjx_data: mjx.Data = None
    info: dict = None


class BaseSim:
    mj_model: mujoco.MjModel | None
    viewer: ViewerHandle | None
    headless: bool

    def init_state(self) -> State:
        raise NotImplementedError

    def step(self, state: State, action: np.ndarray) -> State:
        """
        Step the simulation with the given action.
        Args:
            state (State): The current state of the simulation.
            action (np.ndarray): The joint position target for the next step.
        """
        raise NotImplementedError

    def reset(self, state: State, sim_ids: np.ndarray | None) -> State:
        """
        Reset the simulation to the initial state.
        Args:
            state (State): The current state of the simulation.
            sim_ids (int): The ID of the simulation instance to reset.
        """
        raise NotImplementedError

    def view(self, state: State, sim_id: int = 0) -> None:
        """
        View the current state of the simulation.
        Args:
            state (State): The current state of the simulation.
            sim_id (int): The ID of the simulation instance to view.
        """
        raise NotImplementedError

    def close(self):
        if not self.headless and self.viewer.is_running():
            self.viewer.close()
        self.mj_model = None
        self.viewer = None
