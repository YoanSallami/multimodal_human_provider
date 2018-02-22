#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import rospy
import re
import time
import numpy
import argparse
import math
import underworlds
import tf2_ros
from underworlds.helpers.geometry import get_world_transform
from underworlds.tools.loader import ModelLoader
from underworlds.helpers.transformations import translation_matrix, quaternion_matrix, euler_matrix, translation_from_matrix
from multimodal_human_provider.msg import GazeInfoArray
from underworlds.types import Camera, Mesh, MESH, Situation

TF_CACHE_TIME = 5.0
DEFAULT_CLIP_PLANE_NEAR = 0.01
DEFAULT_CLIP_PLANE_FAR = 1000.0
DEFAULT_HORIZONTAL_FOV = 50.0
DEFAULT_ASPECT = 1.33333
LOOK_AT_THRESHOLD = 0.7
MIN_NB_DETECTION = 3
MIN_DIST_DETECTION = 0.2
MAX_HEIGHT = 2.5


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
        self.ros_subscriber = {"gaze": rospy.Subscriber("/wp2/gaze", GazeInfoArray, self.callbackGaze)}
        self.human_cameras_ids = {}
        self.ctx = ctx
        self.human_bodies = {}
        self.target = ctx.worlds[output_world]
        self.target_world_name = output_world
        self.reference_frame = reference_frame
        self.mesh_dir = mesh_dir
        self.human_meshes = {}
        self.human_aabb = {}

        self.nb_gaze_detected = {}
        self.added_human_id = []

        self.detection_time = None
        self.reco_durations = []
        self.record_time = False

        self.robot_name = rospy.get_param("robot_name", "pepper")

        self.already_removed_nodes = []

        nodes_loaded = []

        try:
            nodes_loaded = ModelLoader().load(self.mesh_dir + "face.blend", self.ctx,
                                              world=output_world, root=None, only_meshes=True,
                                              scale=1.0)
        except Exception as e:
            rospy.logwarn("[multimodal_human_provider] Exception occurred with %s : %s" % (self.mesh_dir + "face.blend", str(e)))

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

    def callbackGaze(self, msg):
        nodes_to_update = []

        if msg.data:
            for i, gaze in enumerate(msg.data):
                human_id = gaze.person_id
                track_id = gaze.track_id


                if human_id not in self.nb_gaze_detected:
                    self.nb_gaze_detected[human_id] = 0
                else:
                    self.nb_gaze_detected[human_id] += 1

                if track_id == human_id:
                    self.detection_time = time.time()
                    self.record_time = True
                else:
                    if self.record_time:
                        self.reco_durations.append(time.time() - self.detection_time)
                        self.record_time = False

                if gaze.head_gaze_available and self.nb_gaze_detected[human_id] > MIN_NB_DETECTION:
                    new_node = self.create_human_pov(human_id)
                    if human_id in self.human_cameras_ids:
                        new_node.id = self.human_cameras_ids[human_id]
                    else:
                        self.human_cameras_ids[human_id] = new_node.id

                    t = [gaze.head_gaze.position.x, gaze.head_gaze.position.y, gaze.head_gaze.position.z]
                    q = [gaze.head_gaze.orientation.x, gaze.head_gaze.orientation.y, gaze.head_gaze.orientation.z, gaze.head_gaze.orientation.w]
                    if math.sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2]) < MIN_DIST_DETECTION:
                        continue
                    (trans, rot) = self.lookupTransform(self.reference_frame, msg.header.frame_id, rospy.Time(0))

                    offset = euler_matrix(0, math.radians(90), math.radians(90), 'rxyz')

                    transform = numpy.dot(transformation_matrix(trans, rot), transformation_matrix(t, q))

                    new_node.transformation = numpy.dot(transform, offset)
                    if translation_from_matrix(new_node.transformation)[2] > MAX_HEIGHT:
                        continue
                    self.added_human_id.append(human_id)
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

                    #if gaze.probability_looking_at_robot >= LOOK_AT_THRESHOLD:
                    #    self.target.timeline.start(Situation(desc="lookat(human-%s,%s)" % (str(gaze.person_id), self.robot_name)))

        if nodes_to_update:
            self.target.scene.nodes.update(nodes_to_update)

    def clean_humans(self):
        nodes_to_remove = []

        for node in self.target.scene.nodes:
            if node not in self.already_removed_nodes:
                if re.match("human-", node.name):
                    if time.time() - node.last_update > 5.0:
                        nodes_to_remove.append(node)
                        for child in node.children:
                            nodes_to_remove.append(self.target.scene.nodes[child])

        if nodes_to_remove:
            rospy.logwarn(nodes_to_remove)
            self.already_removed_nodes = nodes_to_remove
            self.target.scene.nodes.remove(nodes_to_remove)

    def run(self):
        while not rospy.is_shutdown():
            pass
        import csv
        with open("/home/ysallami/stat.csv", "w") as csvfile:
            fieldnames = ["human_id", "nb_detection", "is_human"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for human_id, nb_detect in self.nb_gaze_detected.items():
                writer.writerow({"human_id": int(human_id), "nb_detection": int(nb_detect),
                                 "is_human": 1 if human_id in self.added_human_id else 0})
            csvfile.close()

        with open("/home/ysallami/duration_stat.csv", "w") as csvfile2:
            fieldnames = ["reco_durations"]
            writer = csv.DictWriter(csvfile2, fieldnames=fieldnames)
            writer.writeheader()
            for duration in self.reco_durations:
                writer.writerow({"reco_durations": duration})
            csvfile2.close()


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




