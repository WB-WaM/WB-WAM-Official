// Copyright 2023 Unitree. All rights reserved.
#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>
#include <chrono>
#include <thread>
#include <eigen3/Eigen/Dense>

namespace utilities 
{

float clamp(float value, float lower, float upper) 
{
    return std::max(lower, std::min(value, upper));
}

void countdown(int seconds) 
{
    while (seconds > 0) 
    {
        std::cout << "Countdown: " << seconds << " seconds\r"; 
        std::flush(std::cout); 
        std::this_thread::sleep_for(std::chrono::seconds(1)); 
        --seconds;
    }
    std::cout << "Countdown finished!      \n"; 
}

float angle2encoder(float desired_angle,float servo_limit_enoder,Eigen::VectorXf limitation,float direction)
{
    desired_angle = clamp(desired_angle,limitation(0),limitation(1));
    float servo_encoder_resolution = 4096 / (2 * M_PI);
    float joint_range = limitation(1) - limitation(0); 
    float encoder_range = joint_range * (M_PI / 180) * servo_encoder_resolution;
    float desired_encoder = direction * (desired_angle - limitation(0))* (encoder_range / joint_range) + servo_limit_enoder;
    return desired_encoder;
}

float encoder2angle(float current_encoder,float servo_limit_enoder,Eigen::VectorXf limitation,float direction)
{
    float servo_encoder_resolution = 4096 / (2 * M_PI);
    float joint_range = limitation(1) - limitation(0); 
    float encoder_range = joint_range * (M_PI / 180) * servo_encoder_resolution;
    float desired_encoder = direction * (current_encoder - servo_limit_enoder)* ( (joint_range * (M_PI / 180)) / encoder_range) + limitation(0) * (M_PI / 180);
    desired_encoder *= (180 / M_PI);
    return desired_encoder;
}

} // namespace utilities

#endif