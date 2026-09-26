#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "ManusDataPublisher.hpp"

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ManusDataPublisher>());
    rclcpp::shutdown();
    return 0;
}
