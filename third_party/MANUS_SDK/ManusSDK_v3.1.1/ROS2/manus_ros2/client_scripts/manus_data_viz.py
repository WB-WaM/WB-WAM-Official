import sys
import threading
import numpy as np
import open3d as o3d
import rclpy
from loop_rate_limiters import RateLimiter
from manus_ros2_msgs.msg import ManusGlove
from rclpy.node import Node


class GloveViz:
    """Open3D visualization for glove data"""

    def __init__(self, glove_id):
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()

        self.glove_id = glove_id
        self.node_meshes = {}  # Stores spheres for each node
        self.node_positions = {}  # Stores positions for line drawing

        self.frame_mesh_restore_rot_mat = np.eye(3)
        self.frame_mesh = None
        self.frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.line_set = o3d.geometry.LineSet()  # Stores lines between nodes

        self.viz.add_geometry(self.frame_mesh)
        self.viz.add_geometry(self.line_set)  # Add empty line set


class MinimalSubscriber(Node):
    def __init__(self, glove_indexes):
        super().__init__("manus_ros2_client_py")

        self.glove_viz_map = {}

        # Subscribe to glove data topics for each glove_id
        self.sub_poses = []
        for glove_id in glove_indexes:
            topic_name = f"/manus_glove_{glove_id}"
            self.sub_poses.append(self.create_subscription(
                ManusGlove,
                topic_name,
                self.glove_callback,
                20,
            ))

        self.timer = self.create_timer(0.02, self.timer_callback)

    def glove_callback(self, msg: ManusGlove):
        """Callback for glove node poses"""
        if msg.glove_id not in self.glove_viz_map:
            self.glove_viz_map[msg.glove_id] = GloveViz(msg.glove_id)

        glove_viz = self.glove_viz_map[msg.glove_id]

        # Store new node positions for line drawing
        glove_viz.node_positions = {}

        # Update node visualization
        for node in msg.raw_nodes:
            pose = node.pose
            node_id = node.node_id

            # Store node position for line drawing
            glove_viz.node_positions[node_id] = np.array([pose.position.x, pose.position.y, pose.position.z])

            if node_id not in glove_viz.node_meshes:
                # Create a sphere for the node
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                mesh.compute_vertex_normals()  # Enable shading
                #mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color
                glove_viz.node_meshes[node_id] = mesh
                glove_viz.viz.add_geometry(mesh)

            mesh = glove_viz.node_meshes[node_id]

            # Translate to correct position
            mesh.translate(-np.asarray(mesh.get_center()), relative=True)  # Reset position
            mesh.translate([pose.position.x, pose.position.y, pose.position.z], relative=False)

            glove_viz.viz.update_geometry(mesh)

        # Update line connections
        self.update_lines(glove_viz, msg)

    def update_lines(self, glove_viz, msg):
        """Update the lines connecting child and parent nodes"""
        line_points = []  # Stores positions of nodes
        line_indices = []  # Stores index pairs for lines

        for node in msg.raw_nodes:
            node_id = node.node_id
            parent_id = node.parent_node_id 

            if parent_id in glove_viz.node_positions and node_id in glove_viz.node_positions:
                # Get child and parent positions
                child_pos = glove_viz.node_positions[node_id]
                parent_pos = glove_viz.node_positions[parent_id]

                # Store positions for LineSet
                start_idx = len(line_points)
                line_points.append(parent_pos)
                line_points.append(child_pos)
                line_indices.append([start_idx, start_idx + 1])

        # Update the line set
        if line_points:
            glove_viz.line_set.points = o3d.utility.Vector3dVector(line_points)
            glove_viz.line_set.lines = o3d.utility.Vector2iVector(line_indices)
            glove_viz.line_set.paint_uniform_color([0, 0, 0])  # Black lines
            glove_viz.viz.update_geometry(glove_viz.line_set)

    def timer_callback(self):
        for glove_viz in self.glove_viz_map.values():
            glove_viz.viz.poll_events()
            glove_viz.viz.update_renderer()


def spin_node(glove_indexes):
    rclpy.init(args=sys.argv)

    minimal_subscriber = MinimalSubscriber(glove_indexes)
    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


def main():
    glove_indexes = [0, 1]  # Add more glove indexes as needed

    spin_thread = threading.Thread(target=spin_node, args=(glove_indexes,), daemon=True)
    spin_thread.start()

    rate = RateLimiter(frequency=120.0, warn=False)
    while True:
        rate.sleep()


if __name__ == "__main__":
    main()
