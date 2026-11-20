#include "csc376_franky/sim_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

namespace csc376_franky
{

namespace
{

constexpr double kCycle = 0.001;
// Bandwidth of the simulated joint tracking [rad/s].
constexpr double kOmega = 40.0;

using Mat4 = std::array<double, 16>; // column-major, like franka::RobotState::O_T_EE

Mat4 multiply(const Mat4& a, const Mat4& b)
{
    Mat4 c{};
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k)
                c[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
    return c;
}

// Craig's (modified) Denavit-Hartenberg transform: Rx(alpha) Tx(a) Rz(theta) Tz(d).
Mat4 dh(double a, double d, double alpha, double theta)
{
    const double ca = std::cos(alpha), sa = std::sin(alpha);
    const double ct = std::cos(theta), st = std::sin(theta);
    return {ct, st * ca, st * sa, 0.0, -st, ct * ca, ct * sa, 0.0, 0.0, -sa, ca, 0.0, a, -d * sa, d * ca, 1.0};
}

constexpr double kHalfPi = 1.5707963267948966;

// The flange to TCP transform Desk uses for the Franka Hand.
Mat4 handTcp()
{
    return dh(0.0, 0.1034, 0.0, -kHalfPi / 2.0);
}

// Base to Franka Hand TCP: the Panda's kinematics from Franka's documentation,
// then the flange to TCP transform. If joints is given, it receives the frame
// of each joint, which turns about its z axis.
Mat4 pandaForwardKinematics(const Vec7& q, std::array<Mat4, kDof>* joints = nullptr)
{
    constexpr double kA[7] = {0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088};
    constexpr double kD[7] = {0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0};
    constexpr double kAlpha[7] = {0.0, -kHalfPi, kHalfPi, kHalfPi, -kHalfPi, kHalfPi, kHalfPi};
    Mat4 T = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    for (int j = 0; j < kDof; ++j)
    {
        T = multiply(T, dh(kA[j], kD[j], kAlpha[j], q[j]));
        if (joints != nullptr)
            (*joints)[j] = T;
    }
    T = multiply(T, dh(0.0, 0.107, 0.0, 0.0));
    return multiply(T, handTcp());
}

std::chrono::steady_clock::duration cycleDuration(double speed)
{
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(kCycle / speed));
}

} // namespace

std::array<double, 16> pandaPose(const Vec7& q)
{
    return pandaForwardKinematics(q);
}

// The geometric Jacobian: each joint moves the TCP with its axis crossed with
// the lever from the axis to the TCP, and turns it about its axis.
Jacobian pandaJacobian(const Vec7& q)
{
    std::array<Mat4, kDof> joints;
    const Mat4 tcp = pandaForwardKinematics(q, &joints);
    Jacobian J{};
    for (int j = 0; j < kDof; ++j)
    {
        const double* axis = &joints[j][8];
        const double lever[3] = {tcp[12] - joints[j][12], tcp[13] - joints[j][13], tcp[14] - joints[j][14]};
        J[j * 6 + 0] = axis[1] * lever[2] - axis[2] * lever[1];
        J[j * 6 + 1] = axis[2] * lever[0] - axis[0] * lever[2];
        J[j * 6 + 2] = axis[0] * lever[1] - axis[1] * lever[0];
        J[j * 6 + 3] = axis[0];
        J[j * 6 + 4] = axis[1];
        J[j * 6 + 5] = axis[2];
    }
    return J;
}

SimBackend::SimBackend(const Vec7& q_start, double speed) : speed_(speed > 0.0 ? speed : 1.0), q_(q_start) {}

void SimBackend::connect()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!refuse_reason_.empty())
        throw BackendNetworkError(refuse_reason_);
    connected_ = true;
}

void SimBackend::disconnect()
{
    std::lock_guard<std::mutex> lock(mutex_);
    connected_ = false;
}

bool SimBackend::isConnected() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return connected_;
}

Measured SimBackend::measuredLocked() const
{
    Measured m;
    m.q = q_;
    m.dq = dq_;
    m.q_d = has_q_d_ ? q_d_ : q_;
    m.O_T_EE = pandaForwardKinematics(q_);
    m.F_T_EE = handTcp();
    m.robot_mode = mode_;
    m.time = time_;
    m.control_command_success_rate = mode_ == static_cast<int>(RobotMode::Move) ? 1.0 : 0.0;
    return m;
}

void SimBackend::read(const ReadCallback& callback)
{
    const auto cycle = cycleDuration(speed_);
    auto next = std::chrono::steady_clock::now();
    while (true)
    {
        Measured m;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (drop_pending_)
            {
                drop_pending_ = false;
                connected_ = false;
            }
            if (!connected_)
                throw BackendNetworkError("libfranka: UDP receive: Timeout (simulated)");
            time_ += kCycle;
            m = measuredLocked();
        }
        if (!callback(m))
            return;
        next += cycle;
        std::this_thread::sleep_until(next);
    }
}

void SimBackend::control(const ControlCallback& callback)
{
    run(callback, Interface::Torques);
}

void SimBackend::controlJointPositions(const ControlCallback& callback)
{
    run(callback, Interface::JointPositions);
}

void SimBackend::controlJointVelocities(const ControlCallback& callback)
{
    run(callback, Interface::JointVelocities);
}

Jacobian SimBackend::endEffectorJacobian(const Measured& measured) const
{
    return pandaJacobian(measured.q);
}

void SimBackend::run(const ControlCallback& callback, Interface interface)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!connected_)
            throw BackendNetworkError("libfranka: not connected (simulated)");
        if (mode_ != static_cast<int>(RobotMode::Idle))
            throw BackendControlError(std::string("libfranka: Move command rejected: command not possible in the "
                                                  "current mode (\"") +
                                      robotModeName(mode_) + "\")!");
        mode_ = static_cast<int>(RobotMode::Move);
        last_motion_errors_.clear();
        q_d_ = q_;
        has_q_d_ = interface == Interface::JointVelocities;
    }
    const auto cycle = cycleDuration(speed_);
    auto next = std::chrono::steady_clock::now();
    double period = 0.0;
    try
    {
        while (true)
        {
            Measured m;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (drop_pending_)
                {
                    drop_pending_ = false;
                    connected_ = false;
                    throw BackendNetworkError("libfranka: UDP receive: Timeout (simulated)");
                }
                if (reflex_pending_)
                {
                    reflex_pending_ = false;
                    mode_ = static_cast<int>(RobotMode::Reflex);
                    current_errors_ = {"cartesian_reflex"};
                    last_motion_errors_ = {"cartesian_reflex"};
                    throw BackendControlError(
                        "libfranka: Move command aborted: motion aborted by reflex! [\"cartesian_reflex\"]");
                }
                time_ += period;
                m = measuredLocked();
            }
            const Command command = callback(m, period);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                // Like the robot, which refuses a velocity motion that ends while it still moves.
                if (interface == Interface::JointVelocities && command.finished &&
                    std::any_of(command.dq_d.begin(), command.dq_d.end(), [](double v) { return v != 0.0; }))
                {
                    mode_ = static_cast<int>(RobotMode::Reflex);
                    current_errors_ = {"joint_motion_generator_velocity_discontinuity"};
                    last_motion_errors_ = current_errors_;
                    throw BackendControlError("libfranka: Motion finished commanded, but the robot is still moving! "
                                              "[\"joint_motion_generator_velocity_discontinuity\"]");
                }
                for (int j = 0; j < kDof; ++j)
                {
                    // The robot's internal controller follows a joint position command closely,
                    if (interface == Interface::JointPositions)
                    {
                        dq_[j] = (command.q_d[j] - q_[j]) / kCycle;
                        q_[j] = command.q_d[j];
                        continue;
                    }
                    // and a joint velocity command by moving its own position command with it.
                    if (interface == Interface::JointVelocities)
                    {
                        dq_[j] = command.dq_d[j];
                        q_d_[j] += dq_[j] * kCycle;
                        q_[j] += dq_[j] * kCycle;
                        continue;
                    }
                    const double ddq =
                        kOmega * kOmega * (command.q_d[j] - q_[j]) + 2.0 * kOmega * (command.dq_d[j] - dq_[j]);
                    dq_[j] += ddq * kCycle;
                    q_[j] += dq_[j] * kCycle;
                }
            }
            if (command.finished)
                break;
            period = kCycle;
            next += cycle;
            std::this_thread::sleep_until(next);
        }
    }
    catch (...)
    {
        // The robot holds where it is once a motion is cancelled.
        std::lock_guard<std::mutex> lock(mutex_);
        dq_.fill(0.0);
        has_q_d_ = false;
        if (mode_ == static_cast<int>(RobotMode::Move))
            mode_ = static_cast<int>(RobotMode::Idle);
        throw;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    dq_.fill(0.0);
    has_q_d_ = false;
    mode_ = static_cast<int>(RobotMode::Idle);
}

void SimBackend::recover()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!connected_)
        throw BackendNetworkError("libfranka: not connected (simulated)");
    if (mode_ == static_cast<int>(RobotMode::UserStopped))
        throw BackendControlError("libfranka: Automatic recovery command rejected: user stop pressed!");
    mode_ = static_cast<int>(RobotMode::Idle);
    current_errors_.clear();
}

void SimBackend::errors(std::vector<std::string>& current, std::vector<std::string>& last_motion) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    current = current_errors_;
    last_motion = last_motion_errors_;
}

void SimBackend::setJointPositions(const Vec7& q)
{
    std::lock_guard<std::mutex> lock(mutex_);
    q_ = q;
    dq_.fill(0.0);
}

void SimBackend::setRobotMode(RobotMode mode)
{
    std::lock_guard<std::mutex> lock(mutex_);
    mode_ = static_cast<int>(mode);
}

void SimBackend::injectReflex()
{
    std::lock_guard<std::mutex> lock(mutex_);
    reflex_pending_ = true;
}

void SimBackend::injectDisturbance(int joint, double offset)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (joint < 0 || joint >= kDof)
        throw std::out_of_range("joint must be 0 to 6");
    q_[joint] += offset;
}

void SimBackend::dropConnection()
{
    std::lock_guard<std::mutex> lock(mutex_);
    drop_pending_ = true;
}

void SimBackend::refuseConnections(const std::string& reason)
{
    std::lock_guard<std::mutex> lock(mutex_);
    refuse_reason_ = reason;
}

SimGripperBackend::SimGripperBackend(double max_width, double speed)
    : speed_(speed > 0.0 ? speed : 1.0), max_width_(max_width), width_(max_width)
{
}

void SimGripperBackend::connect()
{
    std::lock_guard<std::mutex> lock(mutex_);
    connected_ = true;
}

void SimGripperBackend::disconnect()
{
    std::lock_guard<std::mutex> lock(mutex_);
    connected_ = false;
}

bool SimGripperBackend::isConnected() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return connected_;
}

bool SimGripperBackend::moveTo(double target, double speed, bool stop_at_object)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!connected_)
            throw BackendNetworkError("libfranka gripper: not connected (simulated)");
        if (!(target >= 0.0 && target <= max_width_ + 1e-9) || !(speed > 0.0))
            throw BackendControlError("libfranka gripper: Move command failed: invalid width or speed (simulated)");
        stop_requested_ = false;
        is_grasped_ = false;
    }
    const auto cycle = cycleDuration(speed_);
    auto next = std::chrono::steady_clock::now();
    while (true)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_requested_)
                return false;
            const double goal = stop_at_object && object_width_ >= 0.0 ? std::max(target, object_width_) : target;
            const double step = speed * kCycle;
            if (std::abs(goal - width_) <= step)
            {
                width_ = goal;
                return true;
            }
            width_ += goal > width_ ? step : -step;
            time_ += kCycle;
        }
        next += cycle;
        std::this_thread::sleep_until(next);
    }
}

bool SimGripperBackend::homing()
{
    if (!moveTo(0.0, 0.1, false))
        return false;
    return moveTo(max_width_, 0.1, false);
}

bool SimGripperBackend::move(double width, double speed)
{
    return moveTo(width, speed, false);
}

bool SimGripperBackend::grasp(double width, double speed, double /*force*/, double epsilon_inner, double epsilon_outer)
{
    // Like the real hand: close until the fingers meet the object or each other.
    if (!moveTo(0.0, speed, true))
        return false;
    std::lock_guard<std::mutex> lock(mutex_);
    is_grasped_ = width_ >= width - epsilon_inner && width_ <= width + epsilon_outer;
    return is_grasped_;
}

bool SimGripperBackend::stop()
{
    std::lock_guard<std::mutex> lock(mutex_);
    stop_requested_ = true;
    return true;
}

GripperSample SimGripperBackend::readOnce()
{
    std::this_thread::sleep_for(std::chrono::duration<double>(0.02 / speed_));
    std::lock_guard<std::mutex> lock(mutex_);
    if (!connected_)
        throw BackendNetworkError("libfranka gripper: UDP receive: Timeout (simulated)");
    GripperSample sample;
    sample.width = width_;
    sample.max_width = max_width_;
    sample.is_grasped = is_grasped_;
    sample.temperature = 30.0;
    sample.time = time_;
    return sample;
}

void SimGripperBackend::setObjectWidth(double width)
{
    std::lock_guard<std::mutex> lock(mutex_);
    object_width_ = width;
}

std::unique_ptr<Backend> makeSimBackend(const Vec7& q_start, double speed)
{
    return std::make_unique<SimBackend>(q_start, speed);
}

std::unique_ptr<GripperBackend> makeSimGripperBackend(double max_width, double speed)
{
    return std::make_unique<SimGripperBackend>(max_width, speed);
}

} // namespace csc376_franky
