#include <iostream>
#include <chrono>
#include <thread>
#include "dxl.h"
#include "yaml_parser.h"
#include "utilities.h"


int main(int argc, char** argv)
{
    param::helper(argc, argv);

    YamlParser yamlParam_;
    std::string env_cfg_path = "/home/unitree/g1_comp_servo_service/config/config.yaml";
    yamlParam_.setup(env_cfg_path.c_str());

    auto joints = dxl::Motors<2>(1000000);

    for(int id=0;id<2;id++)
    {
        joints.enable(id);
        joints.set_position_p_gain(id,0);
        joints.set_position_d_gain(id,100);
    }

    utilities::countdown(5); // Start countdown
    joints.sync_get_position();
    std::cout<<"servo 0 position : "<<joints.present_position[0]<<std::endl;
    yamlParam_.WriteToYaml("servo0_calibration",joints.present_position[0]);
    yamlParam_.SaveToFile();  // 保存到同一个文件

    utilities::countdown(2); // Start countdown

    float servo_limit_enoder = yamlParam_.ReadFloatFromYaml("servo0_calibration");

    Eigen::VectorXf joint0_limitation(2);
    joint0_limitation = yamlParam_.ReadVectorFromYaml("joint0",2);

    Eigen::VectorXf direction(2);
    direction = yamlParam_.ReadVectorFromYaml("direction",2);

    joints.set_position_p_gain(0,500);
    joints.set_position_d_gain(0,300);
    float joint0_servo_positon = utilities::angle2encoder(0,servo_limit_enoder,joint0_limitation,direction(0));
    joints.set_position(0, joint0_servo_positon);

    utilities::countdown(5); // 开始倒计时

    joints.sync_get_position();
    std::cout<<"servo 1 position : "<<joints.present_position[1]<<std::endl;

    yamlParam_.WriteToYaml("servo1_calibration",joints.present_position[1]);
    yamlParam_.WriteToYaml("has_calibrate",1);
    yamlParam_.SaveToFile();  // 保存到同一个文件

    return 0;
}