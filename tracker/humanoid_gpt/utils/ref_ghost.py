"""Utilities for overlaying a translucent 'ghost' of a reference pose.

The ghost renderer maintains its own ``MjData`` so we can run forward
kinematics on the reference ``qpos`` independently of the main simulation.
Reference geoms are then injected into any ``MjvScene`` (an offscreen
renderer's scene or the passive viewer's ``user_scn``) with a configurable
RGBA tint.
"""

from __future__ import annotations

import mujoco
import numpy as np


class RefGhostRenderer:
    """Overlay a tinted reference-pose ghost on a MuJoCo scene.

    Typical usage::

        ghost = RefGhostRenderer(mj_model)
        ghost.set_qpos(ref_qpos)  # per frame

        # Offscreen renderer
        renderer.update_scene(mj_data, camera=cam)
        ghost.add_to_scene(renderer.scene)
        frame = renderer.render()

        # Passive viewer
        ghost.reset_scene(viewer.user_scn)
        ghost.add_to_scene(viewer.user_scn)
        viewer.sync()
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        rgba: tuple[float, float, float, float] = (1.0, 0.45, 0.45, 0.22),
        catmask: int | None = None,
    ):
        self.mj_model = mj_model
        self.ref_data = mujoco.MjData(mj_model)
        self.rgba = np.asarray(rgba, dtype=np.float32)

        self._opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self._opt)
        # Only visualize visual-group geoms (typical convention: group 0/1 visual).
        # Leave defaults; callers can tweak ``self._opt`` if needed.
        self._pert = mujoco.MjvPerturb()
        # Dynamic only -> skip static world geoms (floor, walls, skybox).
        self._catmask = (
            int(catmask)
            if catmask is not None
            else int(mujoco.mjtCatBit.mjCAT_DYNAMIC)
        )

    def set_qpos(self, qpos: np.ndarray) -> None:
        """Update reference ``MjData`` via forward kinematics on ``qpos``."""
        nq = self.mj_model.nq
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        self.ref_data.qpos[:nq] = q[:nq]
        self.ref_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.ref_data)

    @staticmethod
    def reset_scene(scene: mujoco.MjvScene) -> None:
        """Clear user-added geoms from a scene (e.g. ``viewer.user_scn``)."""
        scene.ngeom = 0

    def add_to_scene(self, scene: mujoco.MjvScene) -> None:
        """Append tinted reference geoms to an existing scene."""
        prev_n = scene.ngeom
        mujoco.mjv_addGeoms(
            self.mj_model,
            self.ref_data,
            self._opt,
            self._pert,
            self._catmask,
            scene,
        )
        for i in range(prev_n, scene.ngeom):
            g = scene.geoms[i]
            # Drop any material/texture so our RGBA tint is honored.
            g.matid = -1
            g.rgba[:] = self.rgba
            # Soften appearance: reduce specular/emission so it reads as a ghost.
            g.emission = 0.0
            g.specular = 0.15
            g.shininess = 0.0
            g.reflectance = 0.0
            g.transparent = 1
