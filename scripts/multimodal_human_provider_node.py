#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import rospy
import numpy
import argparse
import math
import underworlds
import tf2_ros
from underworlds.helpers.geometry import get_world_transform
from underworlds.tools.loader import ModelLoader
from underworlds.helpers.transformations import translation_matrix, quaternion_matrix, euler_matrix
from multimodal_human_provider.msg import PersonTrackletArray
from underworlds.types import Camera, Mesh, MESH

TF_CACHE_TIME = 5.0
DEFAULT_CLIP_PLANE_NEAR = 0.01
DEFAULT_CLIP_PLANE_FAR = 1000.0
DEFAULT_HORIZONTAL_FOV = 50.0
DEFAULT_ASPECT = 1.33333


# just for convenience
def strip_leading_slash(s):
    return s[1:] if s.startswith("/") else s


# just for convenience
def transformation_matrix(t, q):
    translation_mat = translation_matrix(t)
    rotation_mat = quaternion_matrix(q)
    return numpy.dot(translation_mat, rotation_mat)


class MultimodalHumanProvider(object):
    def __init__(self, ctx, output_world, mesh_dir, reference_frame):
        self.ros_subscriber = {"person_tracklet": rospy.Subscriber("/tracklet", PersonTrackletArray, self.callbackPersonTracklet)}
        self.human_cameras_ids = {}
        self.ctx = ctx
        self.human_bodies = {}
        self.target = ctx.worlds[output_world]
        self.target_world_name = output_world
        self.reference_frame = reference_frame
        self.mesh_dir = mesh_dir
        self.human_meshes = {}
        self.human_aabb = {}

        nodes_loaded = []
        rospy.logwarn(self.mesh_dir + "face.blend")
        try:
            nodes_loaded = ModelLoader().load(self.mesh_dir + "face.blend", self.ctx,
                                              world=output_world, root=None, only_meshes=True,
                                              scale=1.0)
        except Exception as e:
            rospy.logwarn("[multimodal_human_provider] Exception occurred with %s : %s" % (self.mesh_dir + "face.blend", str(e)))

        rospy.logwarn(nodes_loaded)
        for n in nodes_loaded:
            if n.type == MESH:
                self.human_meshes["face"] = n.properties["mesh_ids"]
                self.human_aabb["face"] = n.properties["aabb"]

        self.tfBuffer = tf2_ros.Buffer(rospy.Duration(TF_CACHE_TIME), debug=False)
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def create_human_pov(self, id):
        new_node = Camera(name="human-" + str(id))
        new_node.properties["clipplanenear"] = DEFAULT_CLIP_PLANE_NEAR
        new_node.properties["clipplanefar"] = DEFAULT_CLIP_PLANE_FAR
        new_node.properties["horizontalfov"] = math.radians(DEFAULT_HORIZONTAL_FOV)
        new_node.properties["aspect"] = DEFAULT_ASPECT
        new_node.parent = self.target.scene.rootnode.id
        return new_node

    def getLatestCommonTime(self, source_frame, dest_frame):
        """
        This is here to provide compatibility with tf2 without having a dependency with old tf
            See : /opt/ros/kinetic/lib/python2.7/dist-packages/tf/listener.py
        @param source_frame:
        @param dest_frame:
        @return :
        """
        return self.tfBuffer.get_latest_common_time(strip_leading_slash(source_frame), strip_leading_slash(dest_frame))

    def lookupTransform(self, target_frame, source_frame, time):
        """
        This is here to provide compatibility with tf2 without having a dependency with old tf
            See : /opt/ros/kinetic/lib/python2.7/dist-packages/tf/listener.py
        @param target_frame:
        @param source_frame:
        @param time:
        @return :
        """
        msg = self.tfBuffer.lookup_transform(strip_leading_slash(target_frame), strip_leading_slash(source_frame), time)
        t = msg.transform.translation
        r = msg.transform.rotation
        return [t.x, t.y, t.z], [r.x, r.y, r.z, r.w]

    def callbackPersonTracklet(self, msg):
        nodes_to_update = []

        if msg.data:
            for i, tracklet in enumerate(msg.data):
                human_id = tracklet.tracklet_id

                new_node = self.create_human_pov(human_id)
                if human_id in self.human_cameras_ids:
                    new_node.id = self.human_cameras_ids[human_id]
                else:
                    self.human_cameras_ids[human_id] = new_node.id

                t = [tracklet.head.position.x, tracklet.head.position.y, tracklet.head.position.z]
                q = [tracklet.head.orientation.x, tracklet.head.orientation.y, tracklet.head.orientation.z, tracklet.head.orientation.w]

                (trans, rot) = self.lookupTransform(self.reference_frame, msg.header.frame_id, rospy.Time(0))

                offset = euler_matrix(0, math.radians(90), math.radians(90), 'rxyz')

                transform = numpy.dot(transformation_matrix(trans, rot), transformation_matrix(t, q))

                new_node.transformation = numpy.dot(transform, offset)

                nodes_to_update.append(new_node)

                if human_id not in self.human_bodies:
                    self.human_bodies[human_id] = {}

                if "face" not in self.human_bodies[human_id]:
                    new_node = Mesh(name="human_face-"+str(human_id))
                    new_node.properties["mesh_ids"] = self.human_meshes["face"]
                    new_node.properties["aabb"] = self.human_aabb["face"]
                    new_node.parent = self.human_cameras_ids[human_id]
                    offset = euler_matrix(math.radians(90), math.radians(0), math.radians(90), 'rxyz')
                    new_node.transformation = numpy.dot(new_node.transformation, offset)
                    self.human_bodies[human_id]["face"] = new_node.id
                    nodes_to_update.append(new_node)

        if nodes_to_update:
            self.target.scene.nodes.update(nodes_to_update)
            #rospy.loginfo("[robot_monitor] %s Nodes updated", str(len(nodes_to_update)))

    def run(self):
        rospy.spin()


if __name__ == "__main__":

    sys.argv = [arg for arg in sys.argv if "__name" not in arg and "__log" not in arg]
    sys.argc = len(sys.argv)

    parser = argparse.ArgumentParser(description="Add in the given output world, the nodes from input "
                                                 "world and the robot agent from ROS")
    parser.add_argument("output_world", help="Underworlds output world")
    parser.add_argument("mesh_dir", help="The path used to localize the human meshes")
    parser.add_argument("--reference", default="map", help="The reference frame")
    args = parser.parse_args()

    rospy.init_node('multimodal_human_provider', anonymous=False)

    with underworlds.Context("Multimodal human provider") as ctx:
        MultimodalHumanProvider(ctx, args.output_world, args.mesh_dir, args.reference).run()




