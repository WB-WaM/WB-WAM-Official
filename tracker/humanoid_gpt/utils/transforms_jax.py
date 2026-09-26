import jax
import jax.numpy as jp
import jaxlie
from jax.scipy.spatial.transform import Rotation as R


def wxyz2xyzw(wxyz):
    return wxyz[jp.array([1, 2, 3, 0])]

def quat2yaw(q: jp.ndarray) -> jp.ndarray:
    R = jaxlie.SO3.from_quaternion_xyzw(wxyz2xyzw(q)).as_matrix()
    yaw = jp.arctan2(R[1, 0], R[0, 0])
    return WarpPi(yaw)

def WarpPi(ang):
    return jp.arctan2(jp.sin(ang), jp.cos(ang))

def quat2mat(quat, dtype=jp.float32):
    return R.from_quat(jp.roll(quat, -1, axis=-1)).as_matrix().astype(dtype)

def se3_inv(pose: jp.ndarray) -> jp.ndarray:
    rot = pose[..., :3, :3]
    t = pose[..., :3, 3]
    rot_t = jp.swapaxes(rot, -1, -2)
    t_inv = -(rot_t @ t[..., None])[..., 0]
    inv = jp.eye(4)
    inv = jp.broadcast_to(inv, pose.shape)
    inv = inv.at[..., :3, :3].set(rot_t)
    inv = inv.at[..., :3, 3].set(t_inv)
    return inv

def base2navi(base2world: jax.Array) -> jax.Array:
    x = base2world[:, 0]
    x_proj = x.at[2].set(0.0)
    x_proj /= jp.linalg.norm(x_proj)
    z_axis = jp.array([0.0, 0.0, 1.0])
    y_axis = jp.cross(z_axis, x_proj)
    y_axis /= jp.linalg.norm(y_axis)
    x_axis = jp.cross(y_axis, z_axis)
    return jp.column_stack((x_axis, y_axis, z_axis))

