#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

class HDDSpawner(Node):
    def __init__(self):
        super().__init__('hdd_spawner_node')
        
        # Publisher to MoveIt's planning scene
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        
        # Give MoveIt a second to register the publisher before sending the object
        self.timer = self.create_timer(2.0, self.spawn_hdd)
        self.has_spawned = False
        
        self.get_logger().info("✅ HDD Spawner Node Active. Waiting 2s to inject flat cuboid (rotated 90°)...")

    def spawn_hdd(self):
        if self.has_spawned:
            return

        # 1. Create a Collision Object
        hdd = CollisionObject()
        hdd.id = "hdd_chassis"
        hdd.header.frame_id = "world_world"
        hdd.operation = CollisionObject.ADD

        # 2. Define the Shape (Standard 3.5-inch HDD dimensions in meters)
        # Dimensions: 147mm x 101.6mm x 26.1mm
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.147, 0.1016, 0.0261]
        hdd.primitives.append(box)

        # 3. Define the Pose 
        hdd_pose = Pose()
        hdd_pose.position.x = 0.9
        hdd_pose.position.y = 0.06
        hdd_pose.position.z = 0.965 
        
        # --- FIX: FLAT ORIENTATION + 90 DEGREE Z-ROTATION ---
        # A 90-degree yaw is represented by z=0.7071 and w=0.7071
        hdd_pose.orientation.x = 0.0
        hdd_pose.orientation.y = 0.0
        hdd_pose.orientation.z = 0.7071068
        hdd_pose.orientation.w = 0.7071068
        
        hdd.primitive_poses.append(hdd_pose)

        # 4. Add to Planning Scene and Publish
        scene_msg = PlanningScene()
        scene_msg.is_diff = True
        scene_msg.world.collision_objects.append(hdd)
        
        self.scene_pub.publish(scene_msg)
        self.get_logger().info(f"📦 Spawned rotated '{hdd.id}' at X: {hdd_pose.position.x}, Y: {hdd_pose.position.y}, Z: {hdd_pose.position.z}")
        
        self.has_spawned = True
        self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = HDDSpawner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()