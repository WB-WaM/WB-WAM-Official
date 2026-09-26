"""MuJoCo simulation environment and loop for the G1 (and H1) humanoid robots.

DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
commands, steps physics, and publishes observations back via the SDK bridge.
BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
"""

from collections import OrderedDict
import json
import os
import pathlib
from pathlib import Path
import pickle
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_body_names
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
from gear_sonic.utils.mujoco_sim.robot import Robot

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def validate_replay_support_endpoint(endpoint: str) -> str:
    """Accept only local endpoints for the replay-only simulator actuator."""

    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("replay support endpoint must be non-empty")
    endpoint = endpoint.strip()
    parsed = urlsplit(endpoint)
    if parsed.scheme == "ipc":
        if not parsed.path:
            raise ValueError("ipc replay support endpoint must include a path")
        return endpoint
    if parsed.scheme != "tcp":
        raise ValueError("replay support endpoint must use tcp or ipc")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("tcp replay support endpoint must be loopback-only")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid replay support TCP port") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError("replay support TCP port must be in 1..65535")
    return endpoint


class ReplaySupportServer:
    """Non-blocking, idempotent control plane for managed ElasticBand support."""

    SCHEMA_VERSION = 1

    def __init__(self, endpoint, sim_env, context=None):
        endpoint = validate_replay_support_endpoint(endpoint)
        if sim_env.elastic_band is None:
            raise RuntimeError("managed replay support requires ElasticBand")
        import zmq

        self._zmq = zmq
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 32)
        self._socket.setsockopt(zmq.SNDHWM, 32)
        self._socket.bind(endpoint)
        self.endpoint = endpoint
        self._initialize_protocol_state(sim_env)
        print(f"[INFO] Managed replay support bound at {endpoint}")

    def _initialize_protocol_state(self, sim_env):
        self._sim_env = sim_env
        self._generation = 0
        self._owner_run_id = None
        self._response_cache = OrderedDict()
        self._response_cache_limit = 256

    def _status_response(self, **fields):
        response = {
            "schema_version": self.SCHEMA_VERSION,
            "generation": self._generation,
            "owner_run_id": self._owner_run_id,
            "monotonic_ns": time.monotonic_ns(),
            **self._sim_env.replay_support_status(),
            **fields,
        }
        return response

    def _error_response(
        self,
        *,
        operation="",
        run_id="",
        request_id="",
        error_code,
        error_message,
    ):
        return self._status_response(
            ok=False,
            operation=operation,
            run_id=run_id,
            request_id=request_id,
            error_code=error_code,
            error_message=error_message,
        )

    def _cache_response(self, key, fingerprint, response):
        self._response_cache[key] = (fingerprint, response)
        self._response_cache.move_to_end(key)
        while len(self._response_cache) > self._response_cache_limit:
            self._response_cache.popitem(last=False)
        return response

    def handle_request(self, request):
        if not isinstance(request, dict):
            return self._error_response(
                error_code="invalid_request",
                error_message="request must be a JSON object",
            )
        operation = request.get("operation")
        run_id = request.get("run_id")
        request_id = request.get("request_id")
        for name, value in (
            ("operation", operation),
            ("run_id", run_id),
            ("request_id", request_id),
        ):
            if not isinstance(value, str) or not value:
                return self._error_response(
                    operation=operation if isinstance(operation, str) else "",
                    run_id=run_id if isinstance(run_id, str) else "",
                    request_id=request_id if isinstance(request_id, str) else "",
                    error_code="invalid_request",
                    error_message=f"{name} must be a non-empty string",
                )
        schema_version = request.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != self.SCHEMA_VERSION
        ):
            return self._error_response(
                operation=operation,
                run_id=run_id,
                request_id=request_id,
                error_code="schema_mismatch",
                error_message="unsupported replay support schema version",
            )

        try:
            fingerprint = json.dumps(
                request,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            return self._error_response(
                operation=operation,
                run_id=run_id,
                request_id=request_id,
                error_code="invalid_request",
                error_message=f"request is not canonical JSON: {exc}",
            )
        cache_key = (run_id, request_id)
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            cached_fingerprint, cached_response = cached
            if cached_fingerprint == fingerprint:
                return cached_response
            return self._error_response(
                operation=operation,
                run_id=run_id,
                request_id=request_id,
                error_code="request_id_reused",
                error_message="request_id was reused with different content",
            )

        if operation not in {"status", "hold", "release", "reset_hold"}:
            response = self._error_response(
                operation=operation,
                run_id=run_id,
                request_id=request_id,
                error_code="invalid_operation",
                error_message=f"unsupported operation {operation!r}",
            )
            return self._cache_response(cache_key, fingerprint, response)

        if operation == "status":
            response = self._status_response(
                ok=True,
                operation=operation,
                run_id=run_id,
                request_id=request_id,
            )
            return self._cache_response(cache_key, fingerprint, response)

        requested_generation = request.get("generation")
        if (
            isinstance(requested_generation, bool)
            or not isinstance(requested_generation, int)
            or requested_generation != self._generation
        ):
            response = self._error_response(
                operation=operation,
                run_id=run_id,
                request_id=request_id,
                error_code="generation_mismatch",
                error_message=(
                    f"generation {requested_generation!r} does not match "
                    f"{self._generation}"
                ),
            )
            return self._cache_response(cache_key, fingerprint, response)
        if operation == "release" and self._owner_run_id != run_id:
            response = self._error_response(
                operation=operation,
                run_id=run_id,
                request_id=request_id,
                error_code="owner_mismatch",
                error_message="only the current support owner may release",
            )
            return self._cache_response(cache_key, fingerprint, response)

        try:
            if operation == "release":
                self._sim_env.set_replay_support_enabled(False)
            elif operation == "reset_hold":
                self._sim_env.reset_and_hold_replay_support()
                self._owner_run_id = run_id
            else:
                self._sim_env.set_replay_support_enabled(True)
                self._owner_run_id = run_id
        except Exception as exc:
            response = self._error_response(
                operation=operation,
                run_id=run_id,
                request_id=request_id,
                error_code="operation_failed",
                error_message=f"simulator support operation failed: {exc}",
            )
            return self._cache_response(cache_key, fingerprint, response)
        self._generation += 1
        response = self._status_response(
            ok=True,
            operation=operation,
            run_id=run_id,
            request_id=request_id,
        )
        return self._cache_response(cache_key, fingerprint, response)

    def poll_once(self):
        try:
            raw_request = self._socket.recv(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return False
        try:
            request = json.loads(raw_request.decode("utf-8"))
            response = self.handle_request(request)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            response = self._error_response(
                error_code="invalid_json",
                error_message=str(exc),
            )
        self._socket.send_json(response)
        return True

    def close(self):
        socket = getattr(self, "_socket", None)
        if socket is not None:
            socket.close(0)
            self._socket = None


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)
        self.camera_configs = camera_configs

        if not camera_configs and offscreen and enable_image_publish:
            self.camera_configs = {
                "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
            }

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen
        self.elastic_band = None

        self.init_scene()
        self.last_reward = 0

        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def init_scene(self):
        """Initialize the default robot scene"""
        xml_path = str(pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = self.mj_model.body(self.root_body).id

        self.joint_class_map = self._get_dof_indices_by_class()

        self.perform_sysid_search = self.config.get("perform_sysid_search", False)

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand(
                managed=bool(self.config.get("REPLAY_SUPPORT_ENDPOINT"))
            )
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mj_model.body("pelvis").id

        self.body_joint_index = []
        self.left_hand_index = []
        self.right_hand_index = []
        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)
            elif "left_hand" in name:
                self.left_hand_index.append(i)
            elif "right_hand" in name:
                self.right_hand_index.append(i)

        assert len(self.body_joint_index) == self.robot.NUM_JOINTS
        assert len(self.left_hand_index) == self.robot.NUM_HAND_JOINTS
        assert len(self.right_hand_index) == self.robot.NUM_HAND_JOINTS

        self.body_joint_index = np.array(self.body_joint_index)
        self.left_hand_index = np.array(self.left_hand_index)
        self.right_hand_index = np.array(self.right_hand_index)

        self._place_initial_feet_on_ground()
        if self.elastic_band is not None and self.elastic_band.managed:
            self.capture_replay_support_anchor()

    def _foot_geom_min_z(self) -> float | None:
        min_z = None
        for geom_id in range(self.mj_model.ngeom):
            body_id = int(self.mj_model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if "ankle_roll_link" not in body_name:
                continue

            geom_type = int(self.mj_model.geom_type[geom_id])
            center_z = float(self.mj_data.geom_xpos[geom_id, 2])
            size = self.mj_model.geom_size[geom_id]

            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                bottom_z = center_z - float(size[0])
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                xmat = self.mj_data.geom_xmat[geom_id].reshape(3, 3)
                z_extent = float(np.sum(np.abs(xmat[2, :]) * size))
                bottom_z = center_z - z_extent
            else:
                continue

            min_z = bottom_z if min_z is None else min(min_z, bottom_z)
        return min_z

    def _place_initial_feet_on_ground(self):
        if not self.use_floating_root_link:
            return

        self.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        if self.config.get("INITIALIZE_FEET_ON_GROUND", True):
            foot_min_z = self._foot_geom_min_z()
            if foot_min_z is not None:
                clearance = float(self.config.get("INITIAL_FOOT_CLEARANCE", 0.0))
                self.mj_data.qpos[2] += clearance - foot_min_z
                self.mj_data.qvel[:] = 0.0
                mujoco.mj_forward(self.mj_model, self.mj_data)
                print(
                    "Initialized floating base: "
                    f"root_z={self.mj_data.qpos[2]:.4f}, foot_min_z={clearance:.4f}"
                )

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                        )
                    )
        return body_torques

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body("torso_link").xpos.copy()
        # Reorder quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        root_quat = self.mj_data.body("torso_link").xquat.copy()[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                left_hand_torques[i] = (
                    self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.left_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.left_hand_index[i] + self.qvel_offset - 1]
                    )
                )
                right_hand_torques[i] = (
                    self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.right_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.right_hand_index[i] + self.qvel_offset - 1]
                    )
                )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mj_model.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_index - 1]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_index - 1]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.elastic_band:
            if self.elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        # -1: actuator array is 0-based while joint indices from the model are 1-based
        self.torques[self.body_joint_index - 1] = body_torques
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_index - 1] = hand_torques[: self.num_hand_dof]
            self.torques[self.right_hand_index - 1] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.check_fall()

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            elif "mjcf_name" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["mjcf_name"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def set_replay_support_enabled(self, enabled):
        if self.elastic_band is None:
            raise RuntimeError("ElasticBand is unavailable")
        self.elastic_band.set_enabled(bool(enabled))
        if not enabled:
            self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)

    def capture_replay_support_anchor(self):
        """Hold the reset pose in place instead of hoisting it toward z=1 m."""
        if self.elastic_band is None or not self.elastic_band.managed:
            raise RuntimeError("managed replay ElasticBand is unavailable")
        self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.elastic_band.set_anchor_position(
            self.mj_data.xpos[self.band_attached_link]
        )
        self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)

    def reset_and_hold_replay_support(self):
        self.reset()
        self.capture_replay_support_anchor()
        self.set_replay_support_enabled(True)

    def replay_support_status(self):
        if self.elastic_band is None:
            return {
                "support_available": False,
                "support_active": False,
                "support_force_norm": 0.0,
                "sim_time_s": float(self.mj_data.time),
                "base_height_m": float(self.mj_data.qpos[2]),
                "base_quat_wxyz": self.mj_data.qpos[3:7].astype(float).tolist(),
            }
        force_norm = float(
            np.linalg.norm(self.mj_data.xfrc_applied[self.band_attached_link])
        )
        return {
            "support_available": True,
            "support_active": bool(self.elastic_band.enable),
            "support_force_norm": force_norm,
            "support_anchor_position": self.elastic_band.point.astype(float).tolist(),
            "support_position_error_norm": float(
                np.linalg.norm(
                    self.elastic_band.point
                    - self.mj_data.xpos[self.band_attached_link]
                )
            ),
            "sim_time_s": float(self.mj_data.time),
            "base_height_m": float(self.mj_data.qpos[2]),
            "base_quat_wxyz": self.mj_data.qpos[3:7].astype(float).tolist(),
        }

    def check_fall(self):
        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._place_initial_feet_on_ground()


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(
        self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs
    ):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self._running = True

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        else:
            raise ValueError(
                f"Invalid environment name: {env_name}. "
                f"Only 'default' is supported in this minimal build."
            )

        self.replay_support_server = None
        replay_support_endpoint = self.config.get("REPLAY_SUPPORT_ENDPOINT")
        if replay_support_endpoint:
            self.replay_support_server = ReplaySupportServer(
                replay_support_endpoint, self.sim_env
            )

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0
        ts = time.time()

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                if self.replay_support_server is not None:
                    self.replay_support_server.poll_once()
                self.sim_env.sim_step()
                now = time.time()
                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                # Simple rate limiter (replaces ROS rate)
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        self.sim_env.reset()

    def close(self):
        self._running = False
        try:
            replay_support_server = getattr(self, "replay_support_server", None)
            if replay_support_server is not None:
                replay_support_server.close()
                self.replay_support_server = None
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        self.sim_env.handle_keyboard_button(key)
