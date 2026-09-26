#include "ClientLogging.hpp"
#include "SDKClient.hpp"

/// @brief The entry point and main loop of the application.
/// @param argc standard main argument parameter
/// @param argv standard main argument parameter
/// @return ClientReturnCode_Success if successful,
/// ClientReturnCode_FailedToResizeWindow if the client was not able to resize the console window,
/// ClientReturnCode_FailedToInitialize if the system was not able to initialize the SDK,
/// ClientReturnCode_FailedToFindHosts if no hosts were found,
/// ClientReturnCode_FailedToConnect if the sdk could not connect to Core,
/// ClientReturnCode_UnrecognizedStateEncountered if un unrecognized state was encountered,
/// ClientReturnCode_FailedToShutDown if it failed to shut down the SDK wrapper,
/// ClientReturnCode_FailedToRestart if the SDK was not able to reconnect to Core.
int main(int argc, char* argv[])
{
	ManusSDK::ClientLog::print("Starting SDK client!");
	
	ClientReturnCode t_Result;
	SDKClient t_SDKClient;

	t_Result = t_SDKClient.Initialize();

	if (t_Result != ClientReturnCode::ClientReturnCode_Success)
	{
		t_SDKClient.ShutDown();
		return static_cast<int>(t_Result); // Returning initialise failure state
	}
	ManusSDK::ClientLog::print("SDK client is initialized.");

	t_Result = t_SDKClient.Run();
	if (t_Result != ClientReturnCode::ClientReturnCode_Success)
	{
		t_SDKClient.ShutDown();
		return static_cast<int>(t_Result); // Returning run failure state
	}

	// loop is over. disconnect it all
	ManusSDK::ClientLog::print("SDK client is done, shutting down.");
	t_Result = t_SDKClient.ShutDown();
	return static_cast<int>(t_Result);
}