#pragma once

#include "csc376_franky/backend.hpp"

namespace csc376_franky
{

// The frame a twist is given in. Either way it is the velocity of the end
// effector's origin and the angular velocity of the end effector.
enum class TwistFrame
{
    Base,
    EndEffector,
};

// Turns the end-effector twist a client streams into joint velocities, at the
// control rate (velocity control, for servoing).
//
// The twist followed is the last one given, or zero once that is older than
// servo_timeout, so the robot stops when its client goes quiet. It is scaled
// down to the speed limits and approached with bounded acceleration. The
// damped inverse of the Jacobian turns it into joint velocities. Those are
// scaled down, all together, until no joint is faster than its velocity limit
// or than it can still stop from before its position limit, until the end
// effector can still stop above servo_min_height, and they change, all
// together, no faster than the acceleration limit: the end effector then
// moves the way it was asked to, only slower. Near a singularity, where the
// damped inverse cannot give the twist asked for, the robot slows down until
// what it does instead is small, unless that takes it away from the singularity. After stop the twist goes to zero, and
// the motion is done once the robot is at rest.
//
// step() does not allocate, so it is safe in the 1 kHz control callback.
class TwistFollower
{
public:
    enum class Phase
    {
        Running,
        Stopping,
        Done,
    };

    // Why the robot is not following the twist it was given, as bits of limited().
    enum Limited : unsigned
    {
        // The twist is older than servo_timeout and was replaced by zero.
        Timeout = 1u << 0,
        // The twist was faster than the speed limits.
        Speed = 1u << 1,
        // The end effector is at servo_min_height.
        Height = 1u << 2,
        // A joint would be faster than its velocity limit.
        JointVelocity = 1u << 3,
        // A joint is near its position limit and would move towards it.
        JointPosition = 1u << 4,
        // The robot is near a singularity, where it cannot move the way it was asked to.
        Singularity = 1u << 5,
    };

    explicit TwistFollower(const Limits& limits);

    // The twist to follow from now on.
    void setTarget(const Vec6& twist, TwistFrame frame);
    // The end effector's Jacobian, in the base frame, at joint positions q.
    using JacobianAt = std::function<Jacobian(const Vec7& q)>;

    // Advance by period seconds; jacobian is jacobian_at the measured state.
    void step(double period, const Measured& measured, const Jacobian& jacobian, bool stop_requested,
              const JacobianAt& jacobian_at);

    // The joint velocities to command.
    const Vec7& dqCommand() const { return dq_c_; }
    // The twist being followed, in the base frame.
    const Vec6& twist() const { return twist_; }
    Phase phase() const { return phase_; }
    unsigned limited() const { return limited_; }

private:
    Limits limits_;

    Vec6 target_{};
    TwistFrame frame_ = TwistFrame::EndEffector;
    // Since the target was given [s].
    double target_age_ = 0.0;
    Vec6 twist_{};
    Vec7 dq_c_{};
    unsigned limited_ = 0;
    // How long the command has been zero, and the robot at rest, while stopping [s].
    double zero_time_ = 0.0;
    double still_time_ = 0.0;
    Phase phase_ = Phase::Running;
};

// Names of the bits set in limited, for the robot's state.
const char* limitedName(TwistFollower::Limited bit);

} // namespace csc376_franky
