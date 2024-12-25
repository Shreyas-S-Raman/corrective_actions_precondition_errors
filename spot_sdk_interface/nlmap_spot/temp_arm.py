#NOTE: velocity can be zero, changing velocity makes motion smoother
#joint_positions[5]: influence the rotation of gripper like wrapping coil around arm axis
#joint_positions[4]: rotation of gripper like pendulum (up and down)
#joint_positions[0]: influences left and right motion (relative to robot facing front) at the base of the arm
#joint_positions[2]: influences up and down motion of elbow joint (after base of arm)
#joint_positions[1]: influences  up and down motion at base of the arm
#joint_positions[5]: influences movement of the "wrist" left and right

import argparse
import sys
import time

from google.protobuf import wrappers_pb2

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, estop_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import duration_to_seconds

def print_feedback(feedback_resp, logger):
    """ Helper function to query for ArmJointMove feedback, and print it to the console.
        Returns the time_to_goal value reported in the feedback """
    joint_move_feedback = feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_joint_move_feedback
    logger.info(f'  planner_status = {joint_move_feedback.planner_status}')
    logger.info(
        f'  time_to_goal = {duration_to_seconds(joint_move_feedback.time_to_goal):.2f} seconds.')

    # Query planned_points to determine target pose of arm
    logger.info('  planned_points:')
    for idx, points in enumerate(joint_move_feedback.planned_points):
        pos = points.position
        pos_str = f'sh0 = {pos.sh0.value:.3f}, sh1 = {pos.sh1.value:.3f}, el0 = {pos.el0.value:.3f}, el1 = {pos.el1.value:.3f}, wr0 = {pos.wr0.value:.3f}, wr1 = {pos.wr1.value:.3f}'
        logger.info(f'    {idx}: {pos_str}')
        print(f'    {idx}: {pos_str}')
    return duration_to_seconds(joint_move_feedback.time_to_goal)

def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = "Robot is estopped. Please use an external E-Stop client, such as the" \
        " estop SDK example, to configure E-Stop."
        robot.logger.error(error_message)
        raise Exception(error_message)

def make_robot_command(arm_joint_traj):
    """ Helper function to create a RobotCommand from an ArmJointTrajectory.
        The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
        filled out to follow the passed in trajectory. """

    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
    arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
    return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)



def main():
    sdk = bosdyn.client.create_standard_sdk('ArmJointMoveClient')
    robot = sdk.create_robot('tusker')
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    verify_estop(robot)

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):

        robot.logger.info("Powering on robot... This may take a several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        # Example 2: Single point trajectory with maximum acceleration/velocity constraints specified such
        # that the solver has to modify the desired points to honor the constraints

        # NOTE: when all values zero the arm moves forward (in arm direction) with no azimuthal angle and no elevation angle


        # sh0 = 0.0 #positive: moves left (rel to forward facing body) azimuthally | negative: moves right (rel to forward facing body) azimuthally 
        # sh1 = 0.0 #positive: moves down (rel to foward facing body) in elevation angle | negative: moves up (rel to forward facing body) in elevetion angle
        # el0 = 0.0 #positive: moves down (rel to foward facing body) in elevation angle | negative: moves fowrad (extension along direction of body) with no elevation or azimuthal angle [could be extension/contraction where +ve=extend and -ve=contract]
        # el1 = 0.0 #positive: extension (maybe) | negative: ???
        # wr0 = 0.0 #positive: gripper faces down | negative: gripper faces up
        # wr1 = -1.0

        # sh0 = 0.0692
        # sh1 = -1.882
        # el0 = 1.652
        # el1 = -0.0691
        # wr0 = 1.622
        # wr1 = 1.550


        sh0 = 0.0692
        sh1 = -1.5
        el0 = 1.652
        el1 = -0.0691
        wr0 = 1.622
        wr1 = 1.550

        max_vel = wrappers_pb2.DoubleValue(value=1)
        max_acc = wrappers_pb2.DoubleValue(value=5)
        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1, time_since_reference_secs=10)
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point],maximum_velocity=max_vel, maximum_acceleration=max_acc)
        # Make a RobotCommand
        command = make_robot_command(arm_joint_traj)

        # Send the request
        cmd_id = command_client.robot_command(command)
        robot.logger.info('Requesting a single point trajectory with unsatisfiable constraints.')

        # Query for feedback
        feedback_resp = command_client.robot_command_feedback(cmd_id)
        robot.logger.info("Feedback for Example 2: planner modifies trajectory")
        time_to_goal = print_feedback(feedback_resp, robot.logger)
        time.sleep(time_to_goal)


        sh0 = 3.14
        sh1 = -1.3
        el0 = 1.3
        el1 = -0.0691
        wr0 = 1.622
        wr1 = 1.550

        max_vel = wrappers_pb2.DoubleValue(value=1)
        max_acc = wrappers_pb2.DoubleValue(value=5)
        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1, time_since_reference_secs=10)
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point],maximum_velocity=max_vel, maximum_acceleration=max_acc)
        # Make a RobotCommand
        command = make_robot_command(arm_joint_traj)

        # Send the request
        cmd_id = command_client.robot_command(command)
        robot.logger.info('Requesting a single point trajectory with unsatisfiable constraints.')

        # Query for feedback
        feedback_resp = command_client.robot_command_feedback(cmd_id)
        robot.logger.info("Feedback for Example 2: planner modifies trajectory")
        time_to_goal = print_feedback(feedback_resp, robot.logger)
        time.sleep(time_to_goal)


        time.sleep(0.75)
        gripper_open = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        cmd_id = command_client.robot_command(gripper_open)
        time.sleep(10.0)




if __name__=='__main__':
    main()