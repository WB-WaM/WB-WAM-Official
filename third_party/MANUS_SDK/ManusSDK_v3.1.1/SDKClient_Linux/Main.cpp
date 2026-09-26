// Set up a Doxygen group.
/** @addtogroup Main
 *  @{
 */

#include "ClientLogging.hpp"
#include "SDKClient.hpp"
#include <pybind11/pybind11.h>
#include <map>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

namespace py = pybind11;


ClientReturnCode t_Result;
SDKClient t_SDKClient;

int init(double timeout_s = 10.0) {
	{
		std::lock_guard<std::mutex> t_OutputLock(output_map_mutex);
		output_map.clear();
	}
	ClientReturnCode t_InitResult = t_SDKClient.Initialize();
	if (t_InitResult != ClientReturnCode::ClientReturnCode_Success) {
		return static_cast<int>(t_InitResult);
	}
	ClientReturnCode t_RunResult = t_SDKClient.Run();
	if (t_RunResult != ClientReturnCode::ClientReturnCode_Success) {
		return static_cast<int>(t_RunResult);
	}
	const auto t_Start = std::chrono::steady_clock::now();
	while (true){
		{
			std::lock_guard<std::mutex> t_OutputLock(output_map_mutex);
			if (!output_map.empty()) {
				return 0;
			}
		}
		if (timeout_s >= 0.0) {
			const auto t_Elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_Start).count();
			if (t_Elapsed >= timeout_s) {
				return 1;
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

py::dict get_latest_state() {
    py::dict py_dict;
	std::lock_guard<std::mutex> t_OutputLock(output_map_mutex);
    for (const auto& pair : output_map) {
        py::list py_list;  // Initialize a py::list for each vector
        for (double value : pair.second) {
            py_list.append(value);  // Append each value from the vector to the py::list
        }
        py_dict[py::str(pair.first)] = py_list;  // Assign the list to the dict with the string key
    }
    return py_dict;
}

py::dict load_calibration(const std::string& path, uint32_t glove_id) {
	py::dict result;
	result["glove_id"] = glove_id;
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		result["ok"] = false;
		result["error"] = "failed_to_open";
		return result;
	}
	std::vector<unsigned char> bytes(
		(std::istreambuf_iterator<char>(file)),
		std::istreambuf_iterator<char>()
	);
	if (bytes.empty()) {
		result["ok"] = false;
		result["error"] = "empty_file";
		return result;
	}
	SetGloveCalibrationReturnCode t_SetResult = SetGloveCalibrationReturnCode::SetGloveCalibrationReturnCode_Error;
	const SDKReturnCode t_SdkResult = CoreSdk_SetGloveCalibration(glove_id, bytes.data(), static_cast<uint32_t>(bytes.size()), &t_SetResult);
	result["ok"] = (
		t_SdkResult == SDKReturnCode::SDKReturnCode_Success
		&& t_SetResult == SetGloveCalibrationReturnCode::SetGloveCalibrationReturnCode_Success
	);
	result["sdk_return"] = static_cast<int>(t_SdkResult);
	result["set_return"] = static_cast<int>(t_SetResult);
	result["bytes"] = static_cast<int>(bytes.size());
	return result;
}

int shutdown() {
    t_SDKClient.ShutDown();
	{
		std::lock_guard<std::mutex> t_OutputLock(output_map_mutex);
		output_map.clear();
	}
    std::cout<<"Manus shutdown\n";
    return 0;
}

PYBIND11_MODULE(ManusServer, m) {
    m.def("init", &init, py::arg("timeout_s") = 10.0);
	m.def("get_latest_state", &get_latest_state);
	m.def("load_calibration", &load_calibration, py::arg("path"), py::arg("glove_id"));
    m.def("shutdown", &shutdown);
}