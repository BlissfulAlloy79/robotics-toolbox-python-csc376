#include "csc376_franky/twist_follower.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace csc376_franky
{

namespace
{

// How long the command must have been zero, and every joint slower than
// settle_velocity, before the motion ends [s]. The robot refuses a motion
// that ends while it still moves, and libfranka's low-pass filter needs a few
// cycles to bring the velocity it sends down to the zero it is given.
constexpr double kStillTime = 0.05;
// The commanded joint velocities change no faster than this fraction of what
// a trajectory may use on the slowest joint (acceleration_scale x the least of
// kJointAccelerationMax): one value for all joints, which change speed together.
constexpr double kAccelerationFraction = 0.5;
// Slowing down before a joint limit, or before the minimum height, takes up to
// this fraction of that acceleration on top of it.
constexpr double kBrakingFraction = 0.5;
// Near a singularity the damped inverse leaves out part of the twist, so the
// end effector also moves in ways it was not asked to. That part is kept
// below this fraction of the speed limits, by slowing the robot down.
constexpr double kDeviationFraction = 0.1;
// How far ahead along the joint velocities it is checked whether they take the
// robot into the singularity or out of it [rad].
constexpr double kLookAhead = 1e-3;

double norm3(const double* v)
{
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// Moves the three numbers at current towards goal, by at most max_change.
void approach(double* current, const double* goal, double max_change)
{
    const double delta[3] = {goal[0] - current[0], goal[1] - current[1], goal[2] - current[2]};
    const double distance = norm3(delta);
    if (!(distance > max_change))
    {
        std::copy(goal, goal + 3, current);
        return;
    }
    for (int i = 0; i < 3; ++i)
        current[i] += max_change / distance * delta[i];
}

// The speed from which a constant deceleration comes to rest within room, and
// at which the next cycle does not use up more than the room there is.
double brakingSpeed(double deceleration, double room, double period)
{
    if (!(room > 0.0))
        return 0.0;
    const double speed = std::sqrt(2.0 * deceleration * room);
    return period > 0.0 ? std::min(speed, room / period) : speed;
}

// The joint velocities J^T (J J^T + damping^2 I)^-1 twist: those of the
// Jacobian's pseudo-inverse away from a singularity, and bounded near one.
// y is (J J^T + damping^2 I)^-1 twist: the twist that the joint velocities
// give is short of the one asked for by damping^2 y.
Vec7 dampedInverse(const Jacobian& J, const Vec6& twist, double damping, Vec6& y)
{
    y.fill(0.0);
    // A = J J^T + damping^2 I is symmetric and positive definite: solve
    // A y = twist with its Cholesky factor L.
    double A[6][6];
    for (int r = 0; r < 6; ++r)
    {
        for (int c = r; c < 6; ++c)
        {
            double sum = r == c ? damping * damping : 0.0;
            for (int j = 0; j < kDof; ++j)
                sum += J[j * 6 + r] * J[j * 6 + c];
            A[r][c] = A[c][r] = sum;
        }
    }
    double L[6][6] = {};
    for (int r = 0; r < 6; ++r)
    {
        for (int c = 0; c <= r; ++c)
        {
            double sum = A[r][c];
            for (int k = 0; k < c; ++k)
                sum -= L[r][k] * L[c][k];
            if (r != c)
                L[r][c] = sum / L[c][c];
            else if (sum > 0.0)
                L[r][r] = std::sqrt(sum);
            else
                return Vec7{}; // singular without damping, or not a number: do not move
        }
    }
    for (int r = 0; r < 6; ++r)
    {
        double sum = twist[r];
        for (int k = 0; k < r; ++k)
            sum -= L[r][k] * y[k];
        y[r] = sum / L[r][r];
    }
    for (int r = 5; r >= 0; --r)
    {
        double sum = y[r];
        for (int k = r + 1; k < 6; ++k)
            sum -= L[k][r] * y[k];
        y[r] = sum / L[r][r];
    }
    Vec7 dq{};
    for (int j = 0; j < kDof; ++j)
        for (int r = 0; r < 6; ++r)
            dq[j] += J[j * 6 + r] * y[r];
    return dq;
}

} // namespace

const char* limitedName(TwistFollower::Limited bit)
{
    switch (bit)
    {
    case TwistFollower::Timeout: return "timeout";
    case TwistFollower::Speed: return "speed";
    case TwistFollower::Height: return "height";
    case TwistFollower::JointVelocity: return "joint_velocity";
    case TwistFollower::JointPosition: return "joint_position";
    case TwistFollower::Singularity: return "singularity";
    }
    return "unknown";
}

TwistFollower::TwistFollower(const Limits& limits) : limits_(limits) {}

void TwistFollower::setTarget(const Vec6& twist, TwistFrame frame)
{
    target_ = twist;
    frame_ = frame;
    target_age_ = 0.0;
}

void TwistFollower::step(double period, const Measured& measured, const Jacobian& jacobian, bool stop_requested,
                         const JacobianAt& jacobian_at)
{
    if (phase_ == Phase::Done)
        return;
    if (stop_requested && phase_ == Phase::Running)
        phase_ = Phase::Stopping;
    target_age_ += period;
    limited_ = 0;

    // The twist to aim for, in the base frame: zero when stopping, and when
    // the client has gone quiet.
    Vec6 goal{};
    if (phase_ == Phase::Running)
    {
        if (target_age_ > limits_.servo_timeout)
        {
            if (std::any_of(target_.begin(), target_.end(), [](double v) { return v != 0.0; }))
                limited_ |= Timeout;
        }
        else if (frame_ == TwistFrame::EndEffector)
        {
            // O_T_EE is column-major: its first three columns are the end effector's axes.
            const std::array<double, 16>& T = measured.O_T_EE;
            for (int r = 0; r < 3; ++r)
            {
                goal[r] = T[r] * target_[0] + T[4 + r] * target_[1] + T[8 + r] * target_[2];
                goal[3 + r] = T[r] * target_[3] + T[4 + r] * target_[4] + T[8 + r] * target_[5];
            }
        }
        else
        {
            goal = target_;
        }
    }

    // A state that is not a number moves nothing.
    if (!std::all_of(goal.begin(), goal.end(), [](double v) { return std::isfinite(v); }))
        goal.fill(0.0);

    // --- Safety: the speed limits. The whole twist is scaled, so it keeps its direction --- //
    double scale = 1.0;
    const double linear = norm3(goal.data());
    const double angular = norm3(goal.data() + 3);
    if (linear > limits_.servo_linear_velocity)
        scale = std::min(scale, limits_.servo_linear_velocity / linear);
    if (angular > limits_.servo_angular_velocity)
        scale = std::min(scale, limits_.servo_angular_velocity / angular);
    if (scale < 1.0)
    {
        for (double& v : goal)
            v *= scale;
        limited_ |= Speed;
    }

    // The twist reaches its speed limits from rest, and rest from them, in servo_ramp_time.
    const bool ramp = limits_.servo_ramp_time > 0.0;
    const double infinity = std::numeric_limits<double>::infinity();
    const double linear_acceleration = ramp ? limits_.servo_linear_velocity / limits_.servo_ramp_time : infinity;
    const double angular_acceleration = ramp ? limits_.servo_angular_velocity / limits_.servo_ramp_time : infinity;

    // --- Safety: the end effector slows down to rest before the minimum height --- //
    if (goal[2] < 0.0)
    {
        const double room = measured.O_T_EE[14] - limits_.servo_min_height;
        const double fastest = brakingSpeed(kBrakingFraction * linear_acceleration, room, period);
        if (-goal[2] > fastest)
        {
            goal[2] = -fastest;
            limited_ |= Height;
        }
    }

    approach(twist_.data(), goal.data(), ramp ? linear_acceleration * period : infinity);
    approach(twist_.data() + 3, goal.data() + 3, ramp ? angular_acceleration * period : infinity);

    Vec6 y;
    Vec7 dq = dampedInverse(jacobian, twist_, limits_.servo_damping, y);
    if (!std::all_of(dq.begin(), dq.end(), [](double v) { return std::isfinite(v); }))
    {
        dq.fill(0.0);
        y.fill(0.0);
    }

    // --- Safety: near a singularity the robot slows down, so that the end effector hardly moves in --- //
    // --- ways it was not asked to. Here and below all joints are scaled together: slowing one      --- //
    // --- joint alone would send the end effector somewhere it was not asked to go.                 --- //
    scale = 1.0;
    const double damping2 = limits_.servo_damping * limits_.servo_damping;
    // What the end effector does without being asked to, as a fraction of the speed limits.
    const auto deviation = [&](const Vec6& short_of) {
        return std::max(damping2 * norm3(short_of.data()) / limits_.servo_linear_velocity,
                        damping2 * norm3(short_of.data() + 3) / limits_.servo_angular_velocity);
    };
    const double deviates = deviation(y);
    if (deviates > kDeviationFraction)
    {
        // The way out of a singularity is as bent as the way in: the robot only
        // slows down where the twist would be followed worse a little further on.
        double length = 0.0;
        for (int j = 0; j < kDof; ++j)
            length += dq[j] * dq[j];
        length = std::sqrt(length);
        Vec7 ahead = measured.q;
        for (int j = 0; j < kDof; ++j)
            ahead[j] += kLookAhead / length * dq[j];
        Vec6 y_ahead;
        dampedInverse(jacobian_at(ahead), twist_, limits_.servo_damping, y_ahead);
        if (!(deviation(y_ahead) < deviates))
        {
            scale = kDeviationFraction / deviates;
            limited_ |= Singularity;
        }
    }

    // --- Safety: the joint velocity limits --- //
    for (int j = 0; j < kDof; ++j)
    {
        const double speed = scale * std::abs(dq[j]);
        const double fastest = limits_.velocity_scale * kJointVelocityMax[j];
        if (speed > fastest)
        {
            scale *= fastest / speed;
            limited_ |= JointVelocity;
        }
    }

    // --- Safety: no joint velocity changes faster than the acceleration limit. The joints approach --- //
    // --- their new velocities together.                                                            --- //
    const double acceleration = kAccelerationFraction * limits_.acceleration_scale *
                                *std::min_element(kJointAccelerationMax.begin(), kJointAccelerationMax.end());
    double change = 0.0;
    for (int j = 0; j < kDof; ++j)
    {
        dq[j] *= scale;
        change = std::max(change, std::abs(dq[j] - dq_c_[j]));
    }
    Vec7 next = dq;
    if (change > acceleration * period)
    {
        for (int j = 0; j < kDof; ++j)
            next[j] = dq_c_[j] + acceleration * period / change * (dq[j] - dq_c_[j]);
    }

    // --- Safety: the robot slows down to rest before a joint's position limit, and before the      --- //
    // --- minimum height, whatever makes the end effector go down. It is the velocities about to be --- //
    // --- commanded that are scaled, to where the fastest joint can still slow down to rest in the  --- //
    // --- room that is left.                                                                        --- //
    double fastest_joint = 0.0;
    double descent = 0.0;
    for (int j = 0; j < kDof; ++j)
    {
        fastest_joint = std::max(fastest_joint, std::abs(next[j]));
        descent -= jacobian[j * 6 + 2] * next[j];
    }
    scale = 1.0;
    const auto brake = [&](double rate, double room, unsigned bit) {
        if (!(rate > 0.0))
            return;
        double allowed = 0.0;
        if (room > 0.0)
        {
            allowed = std::sqrt(2.0 * kBrakingFraction * acceleration * room / (rate * fastest_joint));
            if (period > 0.0)
                allowed = std::min(allowed, room / (rate * period));
        }
        if (allowed < scale)
        {
            scale = allowed;
            limited_ |= bit;
        }
    };
    for (int j = 0; j < kDof; ++j)
    {
        const double room = next[j] > 0.0 ? kJointPositionMax[j] - limits_.servo_joint_limit_margin - measured.q[j]
                                          : measured.q[j] - kJointPositionMin[j] - limits_.servo_joint_limit_margin;
        brake(std::abs(next[j]), room, JointPosition);
    }
    brake(descent, measured.O_T_EE[14] - limits_.servo_min_height, Height);

    // Braking adds to the acceleration, but never more than its share.
    change = 0.0;
    for (int j = 0; j < kDof; ++j)
    {
        next[j] *= scale;
        change = std::max(change, std::abs(next[j] - dq_c_[j]));
    }
    const double most = (1.0 + kBrakingFraction) * acceleration * period;
    if (change > most)
    {
        for (int j = 0; j < kDof; ++j)
            next[j] = dq_c_[j] + most / change * (next[j] - dq_c_[j]);
    }
    dq_c_ = next;

    if (phase_ == Phase::Stopping)
    {
        const bool zero = std::all_of(dq_c_.begin(), dq_c_.end(), [](double v) { return v == 0.0; });
        zero_time_ = zero ? zero_time_ + period : 0.0;
        double fastest = 0.0;
        for (int j = 0; j < kDof; ++j)
            fastest = std::max(fastest, std::abs(measured.dq[j]));
        still_time_ = fastest < limits_.settle_velocity ? still_time_ + period : 0.0;
        if (zero_time_ >= kStillTime && (still_time_ >= kStillTime || zero_time_ >= limits_.settle_timeout))
            phase_ = Phase::Done;
    }
}

} // namespace csc376_franky
