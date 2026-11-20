import time

import numpy as np

from csc376_franky import FrankaJointTrajectoryController


def main():
    np.set_printoptions(precision=4, suppress=True)

    # The twist for the end effector to follow: 1 cm/s straight up, in the robot's base frame
    twist = [0.0, 0.0, 0.01, 0.0, 0.0, 0.0]
    duration = 3.0

    csc376_franky_robot = (
        FrankaJointTrajectoryController()
    )  # The robot server on this station's miniPC

    print("se3_start", csc376_franky_robot.get_current_pose())

    yes_or_else = input(
        "To move the real robot 3 cm up, type [yes], then press enter\n"
    )
    if yes_or_else != "yes":
        print("User did not type [yes], will not move the real robot")
        return

    # Under velocity control the robot follows the twist it was last sent, and
    # stops when no new one has come for a quarter of a second: so the twist is
    # sent again and again, as a program that computes it from a camera would
    with csc376_franky_robot.start_servo() as servo:
        start = time.monotonic()
        while time.monotonic() - start < duration:
            servo.set_twist(twist, frame="base")
            if servo.limited:
                print("The miniPC holds the robot back:", servo.limited)
            time.sleep(0.02)
    # Leaving the with block has slowed the robot down to rest

    print("se3_end", csc376_franky_robot.get_current_pose())


if __name__ == "__main__":
    main()
