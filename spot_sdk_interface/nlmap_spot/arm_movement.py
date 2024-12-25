import argparse
import math
import sys
import time

import numpy as np

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import basic_command_pb2, robot_command_pb2
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_duration
from bosdyn.client import ResponseError, RpcError



sdk = bosdyn.client.create_standard_sdk('ArmTest')
robot = sdk.create_robot('138.16.161.21')

try:
    bosdyn.client.util.authenticate(robot)
    robot.start_time_sync(1)
except RpcError as err:
    LOGGER.error("Failed to communicate with robot: %s" % err)
    print("Failed to communicate with robot: %s" % err)
    exit()

bosdyn.client.util.authenticate(robot)
robot.time_sync.wait_for_sync()
assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
            "such as the estop SDK example, to configure E-Stop."


robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)

frame_name = VISION_FRAME_NAME
DURATION =  10

robot.power_on(timeout_sec=20)
assert robot.is_powered_on(), "Robot power on failed."

with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):

    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    blocking_stand(command_client, timeout_sec=10)

    unstow = RobotCommandBuilder.arm_ready_command()

    robot_state = robot_state_client.get_robot_state()
    vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

    command = robot_command_pb2.RobotCommand()

    num_points = 2
    points = [(1.0, 0, 0.5), (-0.2,0,0.5)]

    for i in range(num_points):

        (x,y,z) = points[i]

        # Using the transform we got earlier, transform the points into the vision frame
        x_ewrt_vision, y_ewrt_vision, z_ewrt_vision = vision_T_body.transform_point(x,y,z)

        # Add a new point to the robot command's arm cartesian command se3 trajectory
        # This will be an se3 trajectory point
        point = command.synchronized_command.arm_command.arm_cartesian_command.pose_trajectory_in_task.points.add(
        )

        # Populate this point with the desired position, rotation, and duration information
        point.pose.position.x = x_ewrt_vision
        point.pose.position.y = y_ewrt_vision
        point.pose.position.z = z_ewrt_vision

        point.pose.rotation.x = vision_T_body.rot.x
        point.pose.rotation.y = vision_T_body.rot.y
        point.pose.rotation.z = vision_T_body.rot.z
        point.pose.rotation.w = vision_T_body.rot.w

        if i==1:
            DURATION = 20
        duration = seconds_to_duration(DURATION)
        point.time_since_reference.CopyFrom(duration)

    command.synchronized_command.arm_command.arm_cartesian_command.root_frame_name = frame_name

    # speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(linear=Vec2(x=2, y=2), angular=0), min_vel=SE2Velocity(linear=Vec2(x=-2, y=-2), angular=0))

    # mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit)

    # command.synchronized_command.mobility_command.params.CopyFrom(RobotCommandBuilder._to_any(mobility_params))

    command_client.robot_command(command, end_time_secs=time.time() + DURATION)
    time.sleep(DURATION + 2)

    stow = RobotCommandBuilder.arm_stow_command()
    stow_command_id = command_client.robot_command(stow)
    block_until_arm_arrives(command_client, stow_command_id, 5)