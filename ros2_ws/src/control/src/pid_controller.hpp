#pragma once

#include <chrono>

class PIDController {
public:
    PIDController(double kp, double ki, double kd, double kff = 0.0);
    
    // Compute control output: takes current value, dt, and target
    double compute(double current, double dt, double target);
    
    // Reset integral and previous error
    void reset();
    
    // Set output limits
    void setLimits(double minOutput, double maxOutput);
    
    // Set gains
    void setGains(double newKp, double newKi, double newKd, double newKff);

private:
    double kp, ki, kd, kff;
    double integral;
    double prevError;
    double minOutput, maxOutput;
    bool hasPrevError;
};
