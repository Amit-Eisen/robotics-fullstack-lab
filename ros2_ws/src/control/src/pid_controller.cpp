#include "pid_controller.hpp"
#include <algorithm>

PIDController::PIDController(double kp, double ki, double kd, double kff)
    : kp(kp), ki(ki), kd(kd), kff(kff),
      integral(0.0), prevError(0.0),
      minOutput(-1000), maxOutput(1000),
      hasPrevError(false) {}

double PIDController::compute(double current, double dt, double target)
{
    if (dt <= 0.0)
    {
        return 0.0;
    }

    double error = target - current;

    integral += error * dt;

    double derivative = 0.0;

    if (hasPrevError)
    {
        derivative = (error - prevError) / dt;
    }

    prevError = error;
    hasPrevError = true;

    double output = kp * error + ki * integral + kd * derivative + kff * target;

    output = std::clamp(output, minOutput, maxOutput);

    return output;
}

void PIDController::reset()
{
    integral = 0.0;
    prevError = 0.0;
    hasPrevError = false;
}

void PIDController::setLimits(double newMinOutput, double newMaxOutput)
{
    minOutput = newMinOutput;
    maxOutput = newMaxOutput;
}

void PIDController::setGains(double newKp, double newKi, double newKd, double newKff)
{
    kp = newKp;
    ki = newKi;
    kd = newKd;
    kff = newKff;
}