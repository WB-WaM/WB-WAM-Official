#include "ManusDataPublisher.hpp"
#include "ManusSDKTypes.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>

#include "ClientLogging.hpp"

using ManusSDK::ClientLog;
using namespace std::chrono_literals;

ManusDataPublisher *ManusDataPublisher::s_Instance = nullptr;

ManusDataPublisher::ManusDataPublisher() : Node("manus_data_publisher")
{
    if (s_Instance != nullptr)
    {
        throw std::runtime_error("This can only be initialized once.");
    }
    s_Instance = this;

    // Initialize static variables
    m_LastLogTime = std::chrono::steady_clock::now();
    m_PublishCountMap.clear();

    // Timer to publish the data
    m_PublishTimer = create_wall_timer(8.333333ms, [this]
                                       { PublishCallback(); }); // 120Hz

    // initialize client
    ClientLog::print("Starting MANUS Data publisher!");

    auto t_Response = Initialize();
    if (t_Response != ClientReturnCode::ClientReturnCode_Success)
    {
        ClientLog::error("Failed to initialize the SDK. Are you sure the correct ManusSDKLibary is used?");
        throw std::runtime_error("Failed to initialize the SDK. Are you sure the correct ManusSDKLibary is used?");
    }
    ClientLog::print("MANUS Data publisher initialized.");

    // SDK is setup. so now go to main loop of the program.
    // first loop until we get a connection
    m_ConnectionType == ConnectionType::ConnectionType_Integrated
        ? ClientLog::print("MANUS data publisher is running in integrated mode.")
        : ClientLog::print("MANUS data publisher is connecting to MANUS Core.");

    while (Connect() != ClientReturnCode::ClientReturnCode_Success)
    {
        // not yet connected. wait
        ClientLog::print("MANUS data publisher could not connect, trying again in a second.");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    // Handmotion is set to none by default having the raw skeleton data wrist rotation be static
    const SDKReturnCode t_HandMotionResult = CoreSdk_SetRawSkeletonHandMotion(m_HandMotion);
    if (t_HandMotionResult != SDKReturnCode::SDKReturnCode_Success)
    {
        ClientLog::error("Failed to set hand motion mode. The value returned was {}.", (int32_t)t_HandMotionResult);
    }
}

ManusDataPublisher::~ManusDataPublisher()
{
    //Send a haptics command to all haptics devices to zero them out
    for (size_t i = 0; i < m_Landscape->gloveDevices.gloveCount; i++)
    {
        if(m_Landscape->gloveDevices.gloves[i].isHaptics)
        {
            ClientLog::print("Sending zero vibration command to glove {}", m_Landscape->gloveDevices.gloves[i].id);
            float t_ZeroVibration[5] = {0, 0, 0, 0, 0};
            CoreSdk_VibrateFingersForGlove(m_Landscape->gloveDevices.gloves[i].id, t_ZeroVibration);
        }
    }

    // loop is over. disconnect it all
    ClientLog::print("MANUS data publisher is done, shutting down.");
    ShutDown();

    if (m_NodeInfo != nullptr)
    {
        delete[] m_NodeInfo;
        m_NodeInfo = nullptr;
    }

    s_Instance = nullptr;
}

/// @brief Initialize the sample console and the SDK.
/// This function attempts to resize the console window and then proceeds to initialize the SDK's interface.
ClientReturnCode ManusDataPublisher::Initialize()
{
    const ClientReturnCode t_IntializeResult = InitializeSDK();
    if (t_IntializeResult != ClientReturnCode::ClientReturnCode_Success)
    {
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;
    }

    return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief Initialize the sdk, register the callbacks and set the coordinate system.
/// This needs to be done before any of the other SDK functions can be used.
ClientReturnCode ManusDataPublisher::InitializeSDK()
{
    // Invalid connection type detected
    if (m_ConnectionType == ConnectionType::ConnectionType_Invalid ||
        m_ConnectionType == ConnectionType::ClientState_MAX_CLIENT_STATE_SIZE)
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;

    SDKReturnCode t_InitializeResult = SDKReturnCode_Error;
    bool t_Remote = m_ConnectionType != ConnectionType::ConnectionType_Integrated;

    if (m_ConnectionType == ConnectionType::ConnectionType_Integrated)
    {
        t_InitializeResult = CoreSdk_InitializeIntegrated();
    }
    else
    {
        t_InitializeResult = CoreSdk_InitializeCore();
    }

    if (t_InitializeResult != SDKReturnCode::SDKReturnCode_Success)
    {
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;
    }

    const ClientReturnCode t_CallBackResults = RegisterAllCallbacks();
    if (t_CallBackResults != ::ClientReturnCode::ClientReturnCode_Success)
    {
        return t_CallBackResults;
    }

    CoordinateSystemVUH_Init(&m_CoordinateSystem);
    const SDKReturnCode t_CoordinateResult = CoreSdk_InitializeCoordinateSystemWithVUH(m_CoordinateSystem, m_WorldSpace);
    if (t_CoordinateResult != SDKReturnCode::SDKReturnCode_Success)
    {
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;
    }

    return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief When shutting down the application, it's important to clean up after the SDK and call it's shutdown function.
/// this will close all connections to the host, close any threads.
/// after this is called it is expected to exit the client program. If not you would need to reinitalize the SDK.
ClientReturnCode ManusDataPublisher::ShutDown()
{
    const SDKReturnCode t_Result = CoreSdk_ShutDown();
    if (t_Result != SDKReturnCode::SDKReturnCode_Success)
    {
        return ClientReturnCode::ClientReturnCode_FailedToShutDownSDK;
    }

    if (!PlatformSpecificShutdown())
    {
        return ClientReturnCode::ClientReturnCode_FailedPlatformSpecificShutdown;
    }

    return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief Used to register all the stream callbacks.
/// Callbacks that are registered functions that get called when a certain 'event' happens, such as data coming in.
/// All of these are optional, but depending on what data you require you may or may not need all of them. For this example we only implement the raw skeleton data.
ClientReturnCode ManusDataPublisher::RegisterAllCallbacks()
{
    const SDKReturnCode t_RegisterRawSkeletonCallbackResult = CoreSdk_RegisterCallbackForRawSkeletonStream(
        *OnRawSkeletonStreamCallback);
    if (t_RegisterRawSkeletonCallbackResult != SDKReturnCode::SDKReturnCode_Success)
    {
        ClientLog::error(
            "Failed to register callback function for processing raw skeletal data from Manus Core. The value returned was {}.",
            (int32_t)t_RegisterRawSkeletonCallbackResult);
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;
    }

    const SDKReturnCode t_RegisterRawDeviceDataStreamCallbackResult = CoreSdk_RegisterCallbackForRawDeviceDataStream(
        *OnRawDeviceDataStreamCallback);
    if (t_RegisterRawDeviceDataStreamCallbackResult != SDKReturnCode::SDKReturnCode_Success)
    {
        ClientLog::error(
            "Failed to register callback function for processing raw device data from Manus Core. The value returned was {}.",
            (int32_t)t_RegisterRawDeviceDataStreamCallbackResult);
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;
    }

    const SDKReturnCode t_RegisterErgonomicsCallbackResult = CoreSdk_RegisterCallbackForErgonomicsStream(
        *OnErgonomicsStreamCallback);
    if (t_RegisterErgonomicsCallbackResult != SDKReturnCode::SDKReturnCode_Success)
    {
        ClientLog::error(
            "Failed to register callback function for processing ergonomics data from Manus Core. The value returned was {}.",
            (int32_t)t_RegisterErgonomicsCallbackResult);
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;
    }

    const SDKReturnCode t_RegisterLandscapeCallbackResult = CoreSdk_RegisterCallbackForLandscapeStream(
        *OnLandscapeCallback);
    if (t_RegisterLandscapeCallbackResult != SDKReturnCode::SDKReturnCode_Success)
    {
        ClientLog::error(
            "Failed to register callback function for processing landscape data from Manus Core. The value returned was {}.",
            (int32_t)t_RegisterLandscapeCallbackResult);
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;
    }

    return ClientReturnCode::ClientReturnCode_Success;
}

/// Read latest data from the gloves and publish them as ros2 messages.
void ManusDataPublisher::PublishCallback()
{
    // Copy data
    m_RawSkeletonMutex.lock();
    std::map<uint32_t, ClientRawSkeleton> t_GloveDataMap = m_GloveDataMap;
    m_RawSkeletonMutex.unlock();

    m_ErgonomicsMutex.lock();
    std::map<uint32_t, ErgonomicsData> t_ErgonomicsDataMap = m_ErgonomicsDataMap;
    m_ErgonomicsMutex.unlock();

    m_RawSensorDataMutex.lock();
    std::map<uint32_t, RawDeviceData> t_RawSensorDataMap = m_RawSensorDataMap;
    m_RawSensorDataMutex.unlock();

    // Retrieve raw skeleton node info, should be identical for all gloves and is consistant for the used Core version
    if (m_NodeInfo == nullptr && !t_GloveDataMap.empty())
    {

        auto t_GloveData = t_GloveDataMap.begin();

        m_NodeInfo = new NodeInfo[t_GloveData->second.info.nodesCount];
        const SDKReturnCode t_Result = CoreSdk_GetRawSkeletonNodeInfoArray(t_GloveData->first, m_NodeInfo, t_GloveData->second.info.nodesCount);

        if (t_Result != SDKReturnCode::SDKReturnCode_Success)
        {
            ClientLog::error("Failed to get Raw Skeleton Hierarchy. The error given was {}.", (int32_t)t_Result);
            return;
        }
    }

    // Fetch latest landscape
    m_LandscapeMutex.lock();
    if (m_NewLandscape != nullptr)
    {
        delete m_Landscape;
        m_Landscape = m_NewLandscape;
        m_NewLandscape = nullptr;
    }
    m_LandscapeMutex.unlock();

    // Construct message for each glove
    if (m_Landscape == nullptr)
    {
        return;
    }

    static bool s_LicenseErrorShown = false;

    if (!s_LicenseErrorShown)
    {
        if (m_Landscape->gloveDevices.dongleCount == 0)
        {
            return;
        }

        if (m_ConnectionType != ConnectionType::ConnectionType_Integrated)
        {
            if (!m_Landscape->settings.license.sdk)
            {
                ClientLog::error("It looks like you don't have a valid SDK license. Please connect a valid license key.");
                s_LicenseErrorShown = true;
                return;
            }
        }
        else
        {
            if (!m_Landscape->settings.license.integrated)
            {
                ClientLog::error("It looks like you don't have a valid SDK Integrated license. Please connect a valid license key.");
                s_LicenseErrorShown = true;
                return;
            }
        }
    }
    for (size_t i = 0; i < m_Landscape->gloveDevices.gloveCount; i++)
    {
        manus_ros2_msgs::msg::ManusGlove t_Msg;
        t_Msg.glove_id = m_Landscape->gloveDevices.gloves[i].id;
        t_Msg.side = SideToString(m_Landscape->gloveDevices.gloves[i].side);

        if (t_GloveDataMap.find(t_Msg.glove_id) == t_GloveDataMap.end())
        {
            continue;
        }

        ClientRawSkeleton t_RawSkel = t_GloveDataMap[t_Msg.glove_id];
        if (t_RawSkel.info.nodesCount == 0)
            continue;

        t_Msg.raw_node_count = t_RawSkel.info.nodesCount;

        for (const auto &node : t_RawSkel.nodes)
        {

            uint32_t t_NodeInfoIndex = 0;
            for (; t_NodeInfoIndex < t_RawSkel.info.nodesCount; t_NodeInfoIndex++)
            {
                if (m_NodeInfo[t_NodeInfoIndex].nodeId == node.id)
                {
                    break;
                }
            }

            manus_ros2_msgs::msg::ManusRawNode t_Node;
            t_Node.node_id = node.id;
            t_Node.parent_node_id = m_NodeInfo[t_NodeInfoIndex].parentId;
            t_Node.joint_type = JointTypeToString(m_NodeInfo[t_NodeInfoIndex].fingerJointType);
            t_Node.chain_type = ChainTypeToString(m_NodeInfo[t_NodeInfoIndex].chainType);

            ManusVec3 t_Pos = node.transform.position;
            ManusQuaternion t_Rot = node.transform.rotation;

            geometry_msgs::msg::Pose t_Pose;
            t_Pose.position.x = t_Pos.x;
            t_Pose.position.y = t_Pos.y;
            t_Pose.position.z = t_Pos.z;
            t_Pose.orientation.x = t_Rot.x;
            t_Pose.orientation.y = t_Rot.y;
            t_Pose.orientation.z = t_Rot.z;
            t_Pose.orientation.w = t_Rot.w;

            t_Node.pose = t_Pose;

            t_Msg.raw_nodes.push_back(t_Node);
        }

        // Ergonomics data
        if (t_ErgonomicsDataMap.find(t_Msg.glove_id) == t_ErgonomicsDataMap.end())
        {
            ClientLog::error("Ergonomics data not found for glove_id: {}", t_Msg.glove_id);
            continue;
        }

        ErgonomicsData t_ErgoData = t_ErgonomicsDataMap[t_Msg.glove_id];
        t_Msg.ergonomics_count = ErgonomicsDataType_MAX_SIZE / 2;

        for (size_t y = 0; y < ErgonomicsDataType_MAX_SIZE; y++)
        {
            if (ErgonomicsDataTypeToSide(static_cast<ErgonomicsDataType>(y)) != m_Landscape->gloveDevices.gloves[i].side)
                continue;

            manus_ros2_msgs::msg::ManusErgonomics t_ErgoMsg;
            t_ErgoMsg.type = ErgonomicsDataTypeToString(static_cast<ErgonomicsDataType>(y));
            t_ErgoMsg.value = t_ErgoData.data[y];
            t_Msg.ergonomics.push_back(t_ErgoMsg);
        }

        // Raw sensor data
        if (t_RawSensorDataMap.find(t_Msg.glove_id) != t_RawSensorDataMap.end())
        {
            RawDeviceData t_RawSensorData = t_RawSensorDataMap[t_Msg.glove_id];

            if (t_RawSensorData.sensorCount > 0)
            {
                t_Msg.raw_sensor_count = t_RawSensorData.sensorCount;
                t_Msg.raw_sensor.resize(t_RawSensorData.sensorCount);

                geometry_msgs::msg::Quaternion t_Rot;

                t_Rot.x = t_RawSensorData.rotation.x;
                t_Rot.y = t_RawSensorData.rotation.y;
                t_Rot.z = t_RawSensorData.rotation.z;
                t_Rot.w = t_RawSensorData.rotation.w;
                t_Msg.raw_sensor_orientation = t_Rot;

                for (size_t i = 0; i < t_RawSensorData.sensorCount; i++)
                {
                    geometry_msgs::msg::Pose t_Pose;

                    t_Pose.position.x = t_RawSensorData.sensorData[i].position.x;
                    t_Pose.position.y = t_RawSensorData.sensorData[i].position.y;
                    t_Pose.position.z = t_RawSensorData.sensorData[i].position.z;
                    t_Pose.orientation.x = t_RawSensorData.sensorData[i].rotation.x;
                    t_Pose.orientation.y = t_RawSensorData.sensorData[i].rotation.y;
                    t_Pose.orientation.z = t_RawSensorData.sensorData[i].rotation.z;
                    t_Pose.orientation.w = t_RawSensorData.sensorData[i].rotation.w;
                    t_Msg.raw_sensor[i] = t_Pose;
                }
            }
        }

        // Find a publisher for the glove, if not present create one
        auto t_Publisher = m_GlovePublisher.find(t_Msg.glove_id);
        if (t_Publisher == m_GlovePublisher.end())
        {
            std::string topic_name = "manus_glove_" + std::to_string(m_GlovePublisher.size());
            auto t_NewPublisher = this->create_publisher<manus_ros2_msgs::msg::ManusGlove>(topic_name, 10);
            t_Publisher = m_GlovePublisher.emplace(t_Msg.glove_id, t_NewPublisher).first;
            // (Re)create vibration subscribers for all gloves
            UpdateVibrationSubscribers();
        }

        if (t_Publisher->second)
        {
            t_Publisher->second->publish(t_Msg);
        }

        m_PublishCountMap[t_Msg.glove_id]++;

        auto t_Now = std::chrono::steady_clock::now();
        if (t_Now - m_LastLogTime >= std::chrono::seconds(10))
        {
            std::ostringstream t_Oss;
            for (const auto &t_Entry : m_PublishCountMap)
            {
                t_Oss << "Glove ID: " << t_Entry.first << ", publishes in the last 10 seconds: " << t_Entry.second << "\n";
            }

            ClientLog::print(t_Oss.str().c_str());
            m_PublishCountMap.clear();
            m_LastLogTime = t_Now;
        }

    }

}

/// @brief the client will now try to connect to MANUS Core via the SDK when the ConnectionType is not integrated. These steps still need to be followed when using the integrated ConnectionType.
ClientReturnCode ManusDataPublisher::Connect()
{
    SDKReturnCode t_StartResult = CoreSdk_LookForHosts(5, false);
    if (t_StartResult != SDKReturnCode::SDKReturnCode_Success)
    {
        return ClientReturnCode::ClientReturnCode_FailedToFindHosts;
    }

    uint32_t t_NumberOfHostsFound = 0;
    SDKReturnCode t_NumberResult = CoreSdk_GetNumberOfAvailableHostsFound(&t_NumberOfHostsFound);
    if (t_NumberResult != SDKReturnCode::SDKReturnCode_Success)
    {
        return ClientReturnCode::ClientReturnCode_FailedToFindHosts;
    }

    if (t_NumberOfHostsFound == 0)
    {
        return ClientReturnCode::ClientReturnCode_FailedToFindHosts;
    }

    std::unique_ptr<ManusHost[]> t_AvailableHosts;
    t_AvailableHosts.reset(new ManusHost[t_NumberOfHostsFound]);

    SDKReturnCode t_HostsResult = CoreSdk_GetAvailableHostsFound(t_AvailableHosts.get(), t_NumberOfHostsFound);
    if (t_HostsResult != SDKReturnCode::SDKReturnCode_Success)
    {
        return ClientReturnCode::ClientReturnCode_FailedToFindHosts;
    }

    bool t_Autoconnect = m_Ip.empty();

    uint32_t t_HostSelection = 0;
    if (t_Autoconnect && t_NumberOfHostsFound != 0)
    {
        ClientLog::print("Autoconnecting to the first host found.");
    }
    else
    {
        ClientLog::print("Looking for host with IP address: {}", m_Ip);
        for (size_t i = 0; i < t_NumberOfHostsFound; i++)
        {
            auto t_HostInfo = t_AvailableHosts[i];
            std::string t_HostAddr = t_HostInfo.ipAddress;
            std::string hostIp = t_HostAddr.substr(0, t_HostAddr.find(':'));
            if (hostIp == m_Ip)
            {
                t_HostSelection = i;
                break;
            }
        }
    }

    SDKReturnCode t_ConnectResult = CoreSdk_ConnectToHost(t_AvailableHosts[t_HostSelection]);

    if (t_ConnectResult == SDKReturnCode::SDKReturnCode_NotConnected)
    {
        return ClientReturnCode::ClientReturnCode_FailedToConnect;
    }

    ClientLog::print("Manus Core connected.");
    return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief This gets called when the client is connected and there is glove data available.
/// @param p_RawSkeletonStreamInfo contains the meta data on what data is available and needs to be retrieved from the SDK.
/// The data is not directly passed to the callback, but needs to be retrieved from the SDK for it to be used. This is demonstrated in the function below.
void ManusDataPublisher::OnRawSkeletonStreamCallback(const SkeletonStreamInfo *const p_RawSkeletonStreamInfo)
{
    if (s_Instance)
    {

        s_Instance->m_RawSkeletonMutex.lock();
        for (uint32_t i = 0; i < p_RawSkeletonStreamInfo->skeletonsCount; i++)
        {
            ClientRawSkeleton t_NxtClientRawSkeleton;
            CoreSdk_GetRawSkeletonInfo(i, &t_NxtClientRawSkeleton.info);

            // Retrieves info on the skeletonData, like deviceID and the amount of nodes.
            t_NxtClientRawSkeleton.nodes.resize(t_NxtClientRawSkeleton.info.nodesCount);
            t_NxtClientRawSkeleton.info.publishTime = p_RawSkeletonStreamInfo->publishTime;

            // Retrieves the skeletonData, which contains the node data.
            CoreSdk_GetRawSkeletonData(i, t_NxtClientRawSkeleton.nodes.data(), t_NxtClientRawSkeleton.info.nodesCount);

            s_Instance->m_GloveDataMap.insert_or_assign(t_NxtClientRawSkeleton.info.gloveId, t_NxtClientRawSkeleton);
        }
        s_Instance->m_RawSkeletonMutex.unlock();
    }
}

void ManusDataPublisher::OnRawDeviceDataStreamCallback(const RawDeviceDataInfo *const p_RawDeviceDataInfo)
{
    if (s_Instance)
    {
        std::vector<RawDeviceData> t_NewData;
        t_NewData.resize(p_RawDeviceDataInfo->rawDeviceDataCount);

        s_Instance->m_RawSensorDataMutex.lock();

        for (uint32_t i = 0; i < p_RawDeviceDataInfo->rawDeviceDataCount; i++)
        {
            CoreSdk_GetRawDeviceData(i, &t_NewData[i]);
            s_Instance->m_RawSensorDataMap.insert_or_assign(t_NewData[i].id, t_NewData[i]);
        }

        s_Instance->m_RawSensorDataMutex.unlock();
    }
}

void ManusDataPublisher::OnErgonomicsStreamCallback(const ErgonomicsStream *const p_Ergo)
{
    if (s_Instance)
    {
        for (uint32_t i = 0; i < p_Ergo->dataCount; i++)
        {
            if (p_Ergo->data[i].isUserID)
                continue;

            ErgonomicsData t_Ergo;
            t_Ergo.id = p_Ergo->data[i].id;
            t_Ergo.isUserID = p_Ergo->data[i].isUserID;

            for (int j = 0; j < ErgonomicsDataType::ErgonomicsDataType_MAX_SIZE; j++)
            {
                t_Ergo.data[j] = p_Ergo->data[i].data[j];
            }
            s_Instance->m_ErgonomicsMutex.lock();
            s_Instance->m_ErgonomicsDataMap.insert_or_assign(p_Ergo->data[i].id, t_Ergo);
            s_Instance->m_ErgonomicsMutex.unlock();
        }
    }
}

void ManusDataPublisher::OnLandscapeCallback(const Landscape *const p_Landscape)
{
    if (s_Instance == nullptr)
        return;

    Landscape *t_Landscape = new Landscape(*p_Landscape);
    s_Instance->m_LandscapeMutex.lock();
    if (s_Instance->m_NewLandscape != nullptr)
        delete s_Instance->m_NewLandscape;
    s_Instance->m_NewLandscape = t_Landscape;
    s_Instance->m_NewGestureLandscapeData.resize(t_Landscape->gestureCount);
    CoreSdk_GetGestureLandscapeData(s_Instance->m_NewGestureLandscapeData.data(), (uint32_t)s_Instance->m_NewGestureLandscapeData.size());
    s_Instance->m_LandscapeMutex.unlock();
}

// Helper to (re)create vibration subscribers for all known gloves
void ManusDataPublisher::UpdateVibrationSubscribers()
{
    for (const auto &entry : m_GlovePublisher)
    {
        uint32_t glove_id = entry.first;
        std::string topic_name = "manus_glove_" + std::to_string(std::distance(m_GlovePublisher.begin(), m_GlovePublisher.find(glove_id))) + "/vibration_cmd";
        // Only create if not already present
        if (m_VibrationSubscribers.find(glove_id) == m_VibrationSubscribers.end())
        {
            auto sub = this->create_subscription<manus_ros2_msgs::msg::ManusVibrationCommand>(
                topic_name, 10,
                [this, glove_id](const manus_ros2_msgs::msg::ManusVibrationCommand::SharedPtr msg)
                {
                    this->OnVibrationCommand(msg, glove_id);
                });
            m_VibrationSubscribers[glove_id] = sub;
            ClientLog::print("Subscribed to vibration command topic: {} for glove_id: {}", topic_name.c_str(), glove_id);
        }
    }
}

// Callback for vibration command
void ManusDataPublisher::OnVibrationCommand(const manus_ros2_msgs::msg::ManusVibrationCommand::SharedPtr msg, uint32_t glove_id)
{
    if (!msg){
        return;
    }

    // Clamp intensities to 5 elements and within 0-1 range
    float intensities[5] = {0};
    size_t count = std::min<size_t>(msg->intensities.size(), 5);
    for (size_t i = 0; i < count; ++i)
    {
        intensities[i] = std::clamp(msg->intensities[i], 0.0f, 1.0f);
    }

    GloveLandscapeData t_LandscapeData = GetGloveLandscapeData(glove_id);
    if (!t_LandscapeData.isHaptics)
    {
        // Not a haptics glove or missing in landscape, so no reason to send vibration command
        return;
    }

    SDKReturnCode result = CoreSdk_VibrateFingersForGlove(glove_id, intensities);
    if (result != SDKReturnCode::SDKReturnCode_Success)
    {
        ClientLog::error("Failed to vibrate glove {}: SDK error {}", glove_id, (int)result);
    }
    else
    {
        ClientLog::print("Vibration command sent to glove {}", glove_id);
    }
}

GloveLandscapeData ManusDataPublisher::GetGloveLandscapeData(uint32_t p_GloveID)
{
    std::lock_guard<std::mutex> t_Lock(m_LandscapeMutex);

    if (m_Landscape == nullptr)
    {
        GloveLandscapeData t_Empty;
        return t_Empty;
    }

    for (size_t i = 0; i < m_Landscape->gloveDevices.gloveCount; i++)
    {
        if (m_Landscape->gloveDevices.gloves[i].id == p_GloveID)
        {
            return m_Landscape->gloveDevices.gloves[i];
        }
    }

    GloveLandscapeData t_Empty;
    return t_Empty;
}

std::string ManusDataPublisher::SideToString(Side p_Side)
{
    switch (p_Side)
    {
    case Side_Left:
        return "Left";
    case Side_Right:
        return "Right";
    default:
        return "Invalid";
    }
}

std::string ManusDataPublisher::JointTypeToString(FingerJointType p_FingerJointType)
{
    switch (p_FingerJointType)
    {
    case FingerJointType_Metacarpal:
        return "MCP";
    case FingerJointType_Proximal:
        return "PIP";
    case FingerJointType_Intermediate:
        return "IP";
    case FingerJointType_Distal:
        return "DIP";
    case FingerJointType_Tip:
        return "TIP";
    default:
        return "Invalid";
    }
}

std::string ManusDataPublisher::ChainTypeToString(ChainType p_ChainType)
{
    switch (p_ChainType)
    {
    case ChainType_Arm:
        return "Arm";
    case ChainType_Leg:
        return "Leg";
    case ChainType_Neck:
        return "Neck";
    case ChainType_Spine:
        return "Spine";
    case ChainType_FingerThumb:
        return "Thumb";
    case ChainType_FingerIndex:
        return "Index";
    case ChainType_FingerMiddle:
        return "Middle";
    case ChainType_FingerRing:
        return "Ring";
    case ChainType_FingerPinky:
        return "Pinky";
    case ChainType_Pelvis:
        return "Pelvis";
    case ChainType_Head:
        return "Head";
    case ChainType_Shoulder:
        return "Shoulder";
    case ChainType_Hand:
        return "Hand";
    case ChainType_Foot:
        return "Foot";
    case ChainType_Toe:
        return "Toe";
    default:
        return "Invalid";
    }
}

//-1 for left, 0 for I dunno, 1 for right
Side ManusDataPublisher::ErgonomicsDataTypeToSide(ErgonomicsDataType p_ErgoDataType)
{
    switch (p_ErgoDataType)
    {
    case ErgonomicsDataType_LeftFingerIndexDIPStretch:
    case ErgonomicsDataType_LeftFingerMiddleDIPStretch:
    case ErgonomicsDataType_LeftFingerRingDIPStretch:
    case ErgonomicsDataType_LeftFingerPinkyDIPStretch:
    case ErgonomicsDataType_LeftFingerIndexPIPStretch:
    case ErgonomicsDataType_LeftFingerMiddlePIPStretch:
    case ErgonomicsDataType_LeftFingerRingPIPStretch:
    case ErgonomicsDataType_LeftFingerPinkyPIPStretch:
    case ErgonomicsDataType_LeftFingerIndexMCPStretch:
    case ErgonomicsDataType_LeftFingerMiddleMCPStretch:
    case ErgonomicsDataType_LeftFingerRingMCPStretch:
    case ErgonomicsDataType_LeftFingerPinkyMCPStretch:
    case ErgonomicsDataType_LeftFingerThumbMCPSpread:
    case ErgonomicsDataType_LeftFingerThumbMCPStretch:
    case ErgonomicsDataType_LeftFingerThumbPIPStretch:
    case ErgonomicsDataType_LeftFingerThumbDIPStretch:
    case ErgonomicsDataType_LeftFingerIndexMCPSpread:
    case ErgonomicsDataType_LeftFingerMiddleMCPSpread:
    case ErgonomicsDataType_LeftFingerRingMCPSpread:
    case ErgonomicsDataType_LeftFingerPinkyMCPSpread:
        return Side::Side_Left;
    case ErgonomicsDataType_RightFingerIndexDIPStretch:
    case ErgonomicsDataType_RightFingerMiddleDIPStretch:
    case ErgonomicsDataType_RightFingerRingDIPStretch:
    case ErgonomicsDataType_RightFingerPinkyDIPStretch:
    case ErgonomicsDataType_RightFingerIndexPIPStretch:
    case ErgonomicsDataType_RightFingerMiddlePIPStretch:
    case ErgonomicsDataType_RightFingerRingPIPStretch:
    case ErgonomicsDataType_RightFingerPinkyPIPStretch:
    case ErgonomicsDataType_RightFingerIndexMCPStretch:
    case ErgonomicsDataType_RightFingerMiddleMCPStretch:
    case ErgonomicsDataType_RightFingerRingMCPStretch:
    case ErgonomicsDataType_RightFingerPinkyMCPStretch:
    case ErgonomicsDataType_RightFingerThumbMCPSpread:
    case ErgonomicsDataType_RightFingerThumbMCPStretch:
    case ErgonomicsDataType_RightFingerThumbPIPStretch:
    case ErgonomicsDataType_RightFingerThumbDIPStretch:
    case ErgonomicsDataType_RightFingerIndexMCPSpread:
    case ErgonomicsDataType_RightFingerMiddleMCPSpread:
    case ErgonomicsDataType_RightFingerRingMCPSpread:
    case ErgonomicsDataType_RightFingerPinkyMCPSpread:
        return Side::Side_Right;
    default:
        return Side::Side_Invalid;
    }
}
std::string ManusDataPublisher::ErgonomicsDataTypeToString(ErgonomicsDataType p_ErgoDataType)
{
    switch (p_ErgoDataType)
    {
    case ErgonomicsDataType_LeftFingerIndexDIPStretch:
    case ErgonomicsDataType_RightFingerIndexDIPStretch:
        return "IndexDIPStretch";
    case ErgonomicsDataType_LeftFingerMiddleDIPStretch:
    case ErgonomicsDataType_RightFingerMiddleDIPStretch:
        return "MiddleDIPStretch";
    case ErgonomicsDataType_LeftFingerRingDIPStretch:
    case ErgonomicsDataType_RightFingerRingDIPStretch:
        return "RingDIPStretch";
    case ErgonomicsDataType_LeftFingerPinkyDIPStretch:
    case ErgonomicsDataType_RightFingerPinkyDIPStretch:
        return "PinkyDIPStretch";
    case ErgonomicsDataType_LeftFingerIndexPIPStretch:
    case ErgonomicsDataType_RightFingerIndexPIPStretch:
        return "IndexPIPStretch";
    case ErgonomicsDataType_LeftFingerMiddlePIPStretch:
    case ErgonomicsDataType_RightFingerMiddlePIPStretch:
        return "MiddlePIPStretch";
    case ErgonomicsDataType_LeftFingerRingPIPStretch:
    case ErgonomicsDataType_RightFingerRingPIPStretch:
        return "RingPIPStretch";
    case ErgonomicsDataType_LeftFingerPinkyPIPStretch:
    case ErgonomicsDataType_RightFingerPinkyPIPStretch:
        return "PinkyPIPStretch";
    case ErgonomicsDataType_LeftFingerIndexMCPStretch:
    case ErgonomicsDataType_RightFingerIndexMCPStretch:
        return "IndexMCPStretch";
    case ErgonomicsDataType_LeftFingerMiddleMCPStretch:
    case ErgonomicsDataType_RightFingerMiddleMCPStretch:
        return "MiddleMCPStretch";
    case ErgonomicsDataType_LeftFingerRingMCPStretch:
    case ErgonomicsDataType_RightFingerRingMCPStretch:
        return "RingMCPStretch";
    case ErgonomicsDataType_LeftFingerPinkyMCPStretch:
    case ErgonomicsDataType_RightFingerPinkyMCPStretch:
        return "PinkyMCPStretch";
    case ErgonomicsDataType_LeftFingerThumbMCPSpread:
    case ErgonomicsDataType_RightFingerThumbMCPSpread:
        return "ThumbMCPSpread";
    case ErgonomicsDataType_LeftFingerThumbMCPStretch:
    case ErgonomicsDataType_RightFingerThumbMCPStretch:
        return "ThumbMCPStretch";
    case ErgonomicsDataType_LeftFingerThumbPIPStretch:
    case ErgonomicsDataType_RightFingerThumbPIPStretch:
        return "ThumbPIPStretch";
    case ErgonomicsDataType_LeftFingerThumbDIPStretch:
    case ErgonomicsDataType_RightFingerThumbDIPStretch:
        return "ThumbDIPStretch";
    case ErgonomicsDataType_LeftFingerIndexMCPSpread:
    case ErgonomicsDataType_RightFingerIndexMCPSpread:
        return "IndexSpread";
    case ErgonomicsDataType_LeftFingerMiddleMCPSpread:
    case ErgonomicsDataType_RightFingerMiddleMCPSpread:
        return "MiddleSpread";
    case ErgonomicsDataType_LeftFingerRingMCPSpread:
    case ErgonomicsDataType_RightFingerRingMCPSpread:
        return "RingSpread";
    case ErgonomicsDataType_LeftFingerPinkyMCPSpread:
    case ErgonomicsDataType_RightFingerPinkyMCPSpread:
        return "PinkySpread";
    default:
        return "Invalid";
    }
}
