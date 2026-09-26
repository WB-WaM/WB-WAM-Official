#ifndef _SDK_MINIMAL_CLIENT_HPP_
#define _SDK_MINIMAL_CLIENT_HPP_

#include "ClientPlatformSpecific.hpp"
#include "ManusSDK.h"
#include <mutex>
#include <vector>
#include <memory>
#include <deque>

#include "rclcpp/rclcpp.hpp"
#include "manus_ros2_msgs/msg/manus_ergonomics.hpp"
#include "manus_ros2_msgs/msg/manus_glove.hpp"
#include "manus_ros2_msgs/msg/manus_raw_node.hpp"
#include "manus_ros2_msgs/msg/manus_vibration_command.hpp"

/// @brief The type of connection to core.
enum class ConnectionType : int
{
    ConnectionType_Invalid = 0,
    ConnectionType_Integrated,
    ConnectionType_Local,
    ConnectionType_Remote,
    ClientState_MAX_CLIENT_STATE_SIZE
};

/// @brief Values that can be returned by this application.
enum class ClientReturnCode : int
{
    ClientReturnCode_Success = 0,
    ClientReturnCode_FailedPlatformSpecificInitialization,
    ClientReturnCode_FailedToResizeWindow,
    ClientReturnCode_FailedToInitialize,
    ClientReturnCode_FailedToFindHosts,
    ClientReturnCode_FailedToConnect,
    ClientReturnCode_UnrecognizedStateEncountered,
    ClientReturnCode_FailedToShutDownSDK,
    ClientReturnCode_FailedPlatformSpecificShutdown,
    ClientReturnCode_FailedToRestart,
    ClientReturnCode_FailedWrongTimeToGetData,

    ClientReturnCode_MAX_CLIENT_RETURN_CODE_SIZE
};

/// @brief Used to store the information about the skeleton data coming from the
/// estimation system in Core.
class ClientRawSkeleton
{
public:
    RawSkeletonInfo info;
    std::vector<SkeletonNode> nodes;
};

/// @brief Used to store all the skeleton data coming from the estimation system
/// in Core.
class ClientRawSkeletonCollection
{
public:
    std::vector<ClientRawSkeleton> skeletons;
};

struct GloveRawSkeletonData
{
    GloveRawSkeletonData() = default;

    rclcpp::Publisher<manus_ros2_msgs::msg::ManusGlove>::SharedPtr manusGlovesPub;
};

class ManusDataPublisher : public SDKClientPlatformSpecific, public rclcpp::Node
{
public:
    ManusDataPublisher();

    ~ManusDataPublisher();

    ClientReturnCode Initialize();

    ClientReturnCode InitializeSDK();

    ClientReturnCode ShutDown();

    ClientReturnCode RegisterAllCallbacks();

    static void OnRawSkeletonStreamCallback(
        const SkeletonStreamInfo *const p_RawSkeletonStreamInfo);

    static void OnRawDeviceDataStreamCallback(
        const RawDeviceDataInfo *const p_RawDeviceDataInfo);
    
    static void OnErgonomicsStreamCallback(
        const ErgonomicsStream *const p_ErgonomicsStream);

    static void OnLandscapeCallback(
        const Landscape *const p_LandscapeStream);

    void PublishCallback();

    
    
protected:
    ClientReturnCode Connect();
    
    std::string SideToString(Side p_Side);
    
    std::string JointTypeToString(FingerJointType p_FingerJointType);
    
    std::string ChainTypeToString(ChainType p_ChainType);
    
    Side ErgonomicsDataTypeToSide(ErgonomicsDataType p_ErgonomicsDataType);
    
    std::string ErgonomicsDataTypeToString(ErgonomicsDataType p_ErgonomicsDataType);
    
    // Helper to (re)create vibration subscribers for all known gloves
    void UpdateVibrationSubscribers();
    
    // Callback for vibration command
    void OnVibrationCommand(const manus_ros2_msgs::msg::ManusVibrationCommand::SharedPtr msg, uint32_t glove_id);
    
    GloveLandscapeData GetGloveLandscapeData(uint32_t p_GloveID);

    static ManusDataPublisher *s_Instance;
    
    //Connection type in case of remote m_IP is used, if empty auto discovery is used
    ConnectionType m_ConnectionType = ConnectionType::ConnectionType_Integrated;
    std::string m_Ip = "";
    
    //Coordinate system settings
    bool m_WorldSpace = true;
    CoordinateSystemVUH m_CoordinateSystem = {AxisView::AxisView_XFromViewer, AxisPolarity::AxisPolarity_PositiveZ, Side::Side_Right, 1.0f};
    HandMotion m_HandMotion = HandMotion::HandMotion_None;
    
    std::map<uint32_t, rclcpp::Publisher<manus_ros2_msgs::msg::ManusGlove>::SharedPtr> m_GlovePublisher;
    
    std::mutex m_RawSkeletonMutex;
    std::map<uint32_t, ClientRawSkeleton> m_GloveDataMap;
    NodeInfo* m_NodeInfo = nullptr;
    
    //Add raw sensor data
    std::mutex m_RawSensorDataMutex;
    std::map<uint32_t, RawDeviceData> m_RawSensorDataMap;
    
    //Add ergonomics data
    std::mutex m_ErgonomicsMutex;
    std::map<uint32_t, ErgonomicsData> m_ErgonomicsDataMap;
    
    //Landscape data
    std::mutex m_LandscapeMutex;
    Landscape* m_NewLandscape = nullptr;
    Landscape* m_Landscape = nullptr;
    std::vector<GestureLandscapeData> m_NewGestureLandscapeData;
    std::vector<GestureLandscapeData> m_GestureLandscapeData;
    
    // Vibration command subscribers, mapped by glove_id
    std::map<uint32_t, rclcpp::Subscription<manus_ros2_msgs::msg::ManusVibrationCommand>::SharedPtr> m_VibrationSubscribers;
    
    // MANUS message publishers
    rclcpp::TimerBase::SharedPtr m_PublishTimer;
    
    std::map<uint32_t, int> m_PublishCountMap;
    std::chrono::steady_clock::time_point m_LastLogTime;
};

#endif
