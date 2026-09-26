// SDKMinimalClient.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "SDKMinimalClient.hpp"
#include "ManusSDKTypes.h"
#include <fstream>
#include <iostream>
#include <thread>

#include "ClientLogging.hpp"

using ManusSDK::ClientLog;

SDKMinimalClient* SDKMinimalClient::s_Instance = nullptr;

int main()
{
    ClientLog::print("Starting minimal client!");
    SDKMinimalClient t_Client;
	auto t_Response = t_Client.Initialize();
	if (t_Response != ClientReturnCode::ClientReturnCode_Success)
	{
		ClientLog::error("Failed to initialize the SDK. Are you sure the correct ManusSDKLibary is used?");
		return -1;
	}
    ClientLog::print("minimal client is initialized.");

    // SDK is setup. so now go to main loop of the program.
    t_Client.Run();

    // loop is over. disconnect it all
    ClientLog::print("minimal client is done, shutting down.");
    t_Client.ShutDown();
}

SDKMinimalClient::SDKMinimalClient()
{
	s_Instance = this;
}

SDKMinimalClient::~SDKMinimalClient()
{
	s_Instance = nullptr;
}

/// @brief Initialize the sample console and the SDK.
/// This function attempts to resize the console window and then proceeds to initialize the SDK's interface.
ClientReturnCode SDKMinimalClient::Initialize()
{
	if (!PlatformSpecificInitialization())
	{
		return ClientReturnCode::ClientReturnCode_FailedPlatformSpecificInitialization;
	}

	const ClientReturnCode t_IntializeResult = InitializeSDK();
	if (t_IntializeResult != ClientReturnCode::ClientReturnCode_Success)
	{
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief Initialize the sdk, register the callbacks and set the coordinate system.
/// This needs to be done before any of the other SDK functions can be used.
ClientReturnCode SDKMinimalClient::InitializeSDK()
{
	ClientLog::print("Select what mode you would like to start in (and press enter to submit)");
	ClientLog::print("[1] Core Integrated - This will run standalone without the need for a MANUS Core connection");
	ClientLog::print("[2] Core Local - This will connect to a MANUS Core running locally on your machine");
	ClientLog::print("[3] Core Remote - This will search for a MANUS Core running locally on your network");
	std::string t_ConnectionTypeInput;
	std::cin >> t_ConnectionTypeInput;
	
	switch (t_ConnectionTypeInput[0])
	{
		case '1':
			m_ConnectionType = ConnectionType::ConnectionType_Integrated;
			break;
		case '2':
			m_ConnectionType = ConnectionType::ConnectionType_Local;
			break;
		case '3':
			m_ConnectionType = ConnectionType::ConnectionType_Remote;
			break;
		default:
			m_ConnectionType = ConnectionType::ConnectionType_Invalid;
			ClientLog::print("Invalid input, try again");
			return InitializeSDK();
	}

	// Invalid connection type detected
	if (m_ConnectionType == ConnectionType::ConnectionType_Invalid
		|| m_ConnectionType == ConnectionType::ClientState_MAX_CLIENT_STATE_SIZE)
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;

	// before we can use the SDK, some internal SDK bits need to be initialized.
	SDKReturnCode t_InitializeResult;
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

	// after everything is registered and initialized
	// We specify the coordinate system in which we want to receive the data.
	// (each client can have their own settings. unreal and unity for instance use different coordinate systems)
	// if this is not set, the SDK will not function.
	// The coordinate system used for this example is z-up, x-positive, right-handed and in meter scale.
	CoordinateSystemVUH t_VUH;
	CoordinateSystemVUH_Init(&t_VUH);
	t_VUH.handedness = Side::Side_Right;
	t_VUH.up = AxisPolarity::AxisPolarity_PositiveZ;
	t_VUH.view = AxisView::AxisView_XFromViewer;
	t_VUH.unitScale = 1.0f; //1.0 is meters, 0.01 is cm, 0.001 is mm.

	// The above specified coordinate system is used to initialize and the coordinate space is specified (world vs local).
	const SDKReturnCode t_CoordinateResult = CoreSdk_InitializeCoordinateSystemWithVUH(t_VUH, true);

	/* this is an example of an alternative way of setting up the coordinate system instead of VUH (view, up, handedness)
	CoordinateSystemDirection t_Direction;
	t_Direction.x = AxisDirection::AD_Right;
	t_Direction.y = AxisDirection::AD_Up;
	t_Direction.z = AxisDirection::AD_Forward;
	const SDKReturnCode t_InitializeResult = CoreSdk_InitializeCoordinateSystemWithDirection(t_Direction, true);
	*/

	if (t_CoordinateResult != SDKReturnCode::SDKReturnCode_Success)
	{
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief When shutting down the application, it's important to clean up after the SDK and call it's shutdown function.
/// this will close all connections to the host, close any threads.
/// after this is called it is expected to exit the client program. If not you would need to reinitalize the SDK.
ClientReturnCode SDKMinimalClient::ShutDown()
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
ClientReturnCode SDKMinimalClient::RegisterAllCallbacks()
{
	// Register the callback to receive Raw Skeleton data
	// it is optional, but without it you can not see any resulting skeleton data.
	// see OnRawSkeletonStreamCallback for more details.
	const SDKReturnCode t_RegisterRawSkeletonCallbackResult = CoreSdk_RegisterCallbackForRawSkeletonStream(*OnRawSkeletonStreamCallback);
	if (t_RegisterRawSkeletonCallbackResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to register callback function for processing raw skeletal data from Manus Core. The value returned was {}.", (int32_t)t_RegisterRawSkeletonCallbackResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief main loop
void SDKMinimalClient::Run()
{
	// first loop until we get a connection
	m_ConnectionType == ConnectionType::ConnectionType_Integrated ?
		ClientLog::print("minimal client is running in integrated mode.") :
		ClientLog::print("minimal client is connecting to MANUS Core. (make sure it is running)");

	while (Connect() != ClientReturnCode::ClientReturnCode_Success)
	{
		// not yet connected. wait
		ClientLog::print("minimal client could not connect.trying again in a second.");
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}

	if (m_ConnectionType != ConnectionType::ConnectionType_Integrated)
		ClientLog::print("minimal client is connected, setting up skeletons.");

	// set the hand motion mode of the RawSkeletonStream. This is optional and can be set to any of the HandMotion enum values. Default = None
	// auto will make it move based on available tracking data. If none is available IMU rotation will be used.
	const SDKReturnCode t_HandMotionResult = CoreSdk_SetRawSkeletonHandMotion(HandMotion_Auto);
	if (t_HandMotionResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to set hand motion mode. The value returned was {}.", (int32_t)t_HandMotionResult);
	}

	while (m_Running)
	{
		// check if there is new data available.
		m_RawSkeletonMutex.lock();

		delete m_RawSkeleton;
		m_RawSkeleton = m_NextRawSkeleton;
		m_NextRawSkeleton = nullptr;
		
		m_RawSkeletonMutex.unlock();

		if (m_RawSkeleton != nullptr && m_RawSkeleton->skeletons.size() != 0)
		{
			// print whenever new data is available
			ClientLog::print("raw skeleton data obtained for frame: {}.", std::to_string(m_FrameCounter));
			PrintRawSkeletonNodeInfo();
			m_FrameCounter++;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(33)); // Roughly 30fps, good enough to show the results, but too slow to retrieve all data.
		
		if (GetKeyDown(' ')) // press space to exit
		{
			m_Running = false;
		}
	}
}

void SDKMinimalClient::PrintRawSkeletonNodeInfo()
{
	if ((m_RawSkeleton == nullptr || m_RawSkeleton->skeletons.size() == 0) || m_PrintedNodeInfo)
	{
		if (m_RawSkeleton->skeletons.size() != 0 && m_RawSkeleton->skeletons[0].nodes.size() != 0) {
			
			// prints the position and rotation of the first node in the first skeleton
			ManusVec3 t_Pos = m_RawSkeleton->skeletons[0].nodes[0].transform.position;
			ManusQuaternion t_Rot = m_RawSkeleton->skeletons[0].nodes[0].transform.rotation;

			ClientLog::print("Node 0 Position: x {} y {} z {} Rotation: x {} y {} z {} w {}", t_Pos.x, t_Pos.y, t_Pos.z, t_Rot.x, t_Rot.y, t_Rot.z, t_Rot.w);
		}
		return;
	}

	// this section demonstrates how to interpret the raw skeleton data.
	// how to get the hierarchy of the skeleton, and how to know bone each node represents.

	uint32_t t_GloveId = 0;
	uint32_t t_NodeCount = 0;

	t_GloveId = m_RawSkeleton->skeletons[0].info.gloveId;
	t_NodeCount = 0;

	SDKReturnCode t_Result = CoreSdk_GetRawSkeletonNodeCount(t_GloveId, t_NodeCount);
	if (t_Result != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get Raw Skeleton Node Count. The error given was {}.", (int32_t)t_Result);
		return;
	}

	// now get the hierarchy data, this needs to be used to reconstruct the positions of each node in case the user set up the system with a local coordinate system.
	// having a node position defined as local means that this will be related to its parent. 

	NodeInfo* t_NodeInfo = new NodeInfo[t_NodeCount];
	t_Result = CoreSdk_GetRawSkeletonNodeInfoArray(t_GloveId, t_NodeInfo, t_NodeCount);
	if (t_Result != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get Raw Skeleton Hierarchy. The error given was {}.", (int32_t)t_Result);
		return;
	}

	ClientLog::print("Received Skeleton glove data from Core. skeletons:{} first skeleton glove id:{}", m_RawSkeleton->skeletons.size(), m_RawSkeleton->skeletons[0].info.gloveId);
	ClientLog::print("Printing Node Info:");


	// prints the information for each node, the chain type will which part of the body it is. The finger joint type will be which bone of the finger it is.
	for (size_t i = 0; i < t_NodeCount; i++)
	{
		ClientLog::printWithPadding("Node ID: {} Side: {} ChainType: {} FingerJointType: {}, Parent Node ID: {}",2, std::to_string(t_NodeInfo[i].nodeId), t_NodeInfo[i].side, t_NodeInfo[i].chainType, t_NodeInfo[i].fingerJointType, std::to_string(t_NodeInfo[i].parentId));
	}

	delete[] t_NodeInfo;
	m_PrintedNodeInfo = true;
}

/// @brief the client will now try to connect to MANUS Core via the SDK when the ConnectionType is not integrated. These steps still need to be followed when using the integrated ConnectionType.
ClientReturnCode SDKMinimalClient::Connect()
{
	bool t_ConnectLocally = m_ConnectionType == ConnectionType::ConnectionType_Local;
	SDKReturnCode t_StartResult = CoreSdk_LookForHosts(1, t_ConnectLocally);
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

	uint32_t t_HostSelection = 0;
	if (!t_ConnectLocally && t_NumberOfHostsFound > 1)
	{
		ClientLog::print("Select which host you want to connect to (and press enter to submit)");
		for (size_t i = 0; i < t_NumberOfHostsFound; i++)
		{
			auto t_HostInfo = t_AvailableHosts[i];
			ClientLog::print("[{}] hostname: {} IP address: {}, version {}.{}.{}", i + 1, t_HostInfo.hostName, t_HostInfo.ipAddress, t_HostInfo.manusCoreVersion.major, t_HostInfo.manusCoreVersion.minor, t_HostInfo.manusCoreVersion.patch);
		}
		uint32_t t_HostSelectionInput = 0;
		std::cin >> t_HostSelectionInput;
		if (t_HostSelectionInput <= 0 || t_HostSelectionInput > t_NumberOfHostsFound)
			return ClientReturnCode::ClientReturnCode_FailedToConnect;

		t_HostSelection = t_HostSelectionInput - 1;
	}

	SDKReturnCode t_ConnectResult = CoreSdk_ConnectToHost(t_AvailableHosts[t_HostSelection]);

	if (t_ConnectResult == SDKReturnCode::SDKReturnCode_NotConnected)
	{
		return ClientReturnCode::ClientReturnCode_FailedToConnect;
	}

	return ClientReturnCode::ClientReturnCode_Success;	
}

/// @brief This gets called when the client is connected and there is glove data available.
/// @param p_RawSkeletonStreamInfo contains the meta data on what data is available and needs to be retrieved from the SDK.
/// The data is not directly passed to the callback, but needs to be retrieved from the SDK for it to be used. This is demonstrated in the function below.
void SDKMinimalClient::OnRawSkeletonStreamCallback(const SkeletonStreamInfo* const p_RawSkeletonStreamInfo)
{
	if (s_Instance)
	{
		ClientRawSkeletonCollection* t_NxtClientRawSkeleton = new ClientRawSkeletonCollection();
		t_NxtClientRawSkeleton->skeletons.resize(p_RawSkeletonStreamInfo->skeletonsCount);

		for (uint32_t i = 0; i < p_RawSkeletonStreamInfo->skeletonsCount; i++)
		{
			//Retrieves info on the skeletonData, like deviceID and the amount of nodes.
			CoreSdk_GetRawSkeletonInfo(i, &t_NxtClientRawSkeleton->skeletons[i].info);
			t_NxtClientRawSkeleton->skeletons[i].nodes.resize(t_NxtClientRawSkeleton->skeletons[i].info.nodesCount);
			t_NxtClientRawSkeleton->skeletons[i].info.publishTime = p_RawSkeletonStreamInfo->publishTime;

			//Retrieves the skeletonData, which contains the node data.
			CoreSdk_GetRawSkeletonData(i, t_NxtClientRawSkeleton->skeletons[i].nodes.data(), t_NxtClientRawSkeleton->skeletons[i].info.nodesCount);
		}
		s_Instance->m_RawSkeletonMutex.lock();
		if (s_Instance->m_NextRawSkeleton != nullptr) delete s_Instance->m_NextRawSkeleton;
		s_Instance->m_NextRawSkeleton = t_NxtClientRawSkeleton;
		s_Instance->m_RawSkeletonMutex.unlock();
	}
}


