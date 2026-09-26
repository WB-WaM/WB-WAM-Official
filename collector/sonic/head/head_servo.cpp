// WB-WAM head servo: Unitree G1 two-axis module, Dynamixel Protocol 2.0.
// Source of register/baud/ID layout: vendor/unitree_reference/include/dxl.h.
#include "dynamixel_sdk.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <thread>
#include <unistd.h>

namespace {
volatile std::sig_atomic_t stopped = 0;
void stop(int) { stopped = 1; }
constexpr int baud = 1000000;
constexpr uint16_t torque = 64, goal = 116, present = 132, mode = 11, hardware_error = 70;
constexpr std::array<double, 2> lower{-50, -20}, upper{50, 85}, direction{1, -1};
constexpr double counts_per_degree = 4096.0 / 360.0;
struct Options {
    std::string device;
    bool probe = false, hold = false, target_pose = false, raw = false, teach = false, approved = false;
    double duration = 0;
    std::array<double, 2> degrees{-50, 10};
    std::array<double, 2> calibration{NAN, NAN};
    std::array<double, 2> encoder{3027, 1849};
};
double number(const std::string& value) {
    size_t consumed = 0;
    double result = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(result)) throw std::runtime_error("invalid numeric value");
    return result;
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help") {
            std::cout << "head_servo --device /dev/serial/by-path/... --probe [--duration SECONDS]\n"
                         "head_servo --device PORT --hold-current --motion-approved [--duration SECONDS]\n"
                         "head_servo --device PORT --teach --motion-approved (stdin: hold / stop)\n"
                         "head_servo --device PORT --hold-raw [--joint0-encoder 3027 --joint1-encoder 1849] --motion-approved\n"
                         "head_servo --device PORT --hold-target --servo0-calibration COUNT --servo1-calibration COUNT\n"
                         "  --motion-approved [--joint0-deg -50] [--joint1-deg 10] [--duration SECONDS]\n"
                         "Calibration counts are at each joint's LOWER limit, as in Unitree config.yaml.\n"
                         "Target motion ramps at 10 deg/s; duration is hold time AFTER reaching the pose.\n";
            std::exit(0);
        } else if (a == "--device" && i + 1 < argc) o.device = argv[++i];
        else if (a == "--duration" && i + 1 < argc) {
            o.duration = number(argv[++i]);
        } else if (a == "--probe") o.probe = true;
        else if (a == "--hold-current") o.hold = true;
        else if (a == "--hold-target") o.target_pose = true;
        else if (a == "--hold-raw") o.raw = true;
        else if (a == "--teach") o.teach = true;
        else if (a == "--joint0-encoder" && i + 1 < argc) o.encoder[0] = number(argv[++i]);
        else if (a == "--joint1-encoder" && i + 1 < argc) o.encoder[1] = number(argv[++i]);
        else if (a == "--joint0-deg" && i + 1 < argc) o.degrees[0] = number(argv[++i]);
        else if (a == "--joint1-deg" && i + 1 < argc) o.degrees[1] = number(argv[++i]);
        else if (a == "--servo0-calibration" && i + 1 < argc) o.calibration[0] = number(argv[++i]);
        else if (a == "--servo1-calibration" && i + 1 < argc) o.calibration[1] = number(argv[++i]);
        else if (a == "--motion-approved") o.approved = true;
        else throw std::runtime_error("unknown/incomplete option: " + a);
    }
    if (!std::isfinite(o.duration) || o.duration < 0) throw std::runtime_error("invalid duration");
    if (int(o.probe) + int(o.hold) + int(o.target_pose) + int(o.raw) + int(o.teach) != 1)
        throw std::runtime_error("select exactly one control/probe mode");
    if (!o.probe && !o.approved) throw std::runtime_error("control requires --motion-approved");
    if (o.raw) for (double v : o.encoder)
        if (!std::isfinite(v) || v < 0 || v > 4095 || std::floor(v) != v)
            throw std::runtime_error("raw pose requires two integer encoder counts in 0..4095");
    if (o.target_pose) for (int id = 0; id < 2; ++id) {
        if (!std::isfinite(o.calibration[id]) || o.calibration[id] < 0 || o.calibration[id] > 4095)
            throw std::runtime_error("target pose requires verified servo0/servo1 calibration counts; do not use placeholder zeros");
        if (o.degrees[id] < lower[id] || o.degrees[id] > upper[id])
            throw std::runtime_error("target joint angle outside Unitree limits");
    }
    if (o.device.empty()) throw std::runtime_error("--device is required");
    if (o.probe && o.duration == 0) o.duration = 1;
    return o;
}
class Bus {
    std::unique_ptr<dynamixel::PortHandler> port;
    dynamixel::PacketHandler* packet = dynamixel::PacketHandler::getPacketHandler(2.0);
    int lock_fd = -1;
    std::array<bool, 2> enabled{};
    std::array<bool, 2> changed_gains{};
    std::array<uint16_t, 2> original_p{}, original_d{};
public:
    explicit Bus(const std::string& device) {
        lock_fd = open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (lock_fd < 0) throw std::runtime_error("cannot open serial port; check device and dialout permission");
        if (flock(lock_fd, LOCK_EX | LOCK_NB) != 0 || ioctl(lock_fd, TIOCEXCL) != 0) {
            close(lock_fd); lock_fd = -1;
            throw std::runtime_error("serial port busy");
        }
        // SDK must open the same TTY; advisory flock remains held across that open.
        ioctl(lock_fd, TIOCNXCL);
        port.reset(dynamixel::PortHandler::getPortHandler(device.c_str()));
        if (!port->openPort() || !port->setBaudRate(baud)) {
            close(lock_fd); lock_fd = -1;
            throw std::runtime_error("cannot configure serial port at 1 Mbps");
        }
        if (ioctl(lock_fd, TIOCEXCL) != 0) {
            port->closePort(); close(lock_fd); lock_fd = -1;
            throw std::runtime_error("cannot exclusively claim serial port");
        }
    }
    void check(int rc, uint8_t error, int id) {
        if (rc != COMM_SUCCESS || error) throw std::runtime_error(
            "servo " + std::to_string(id) + ": " + packet->getTxRxResult(rc) +
            " device_error=" + std::to_string(error));
    }
    uint8_t read8(int id, uint16_t addr) {
        uint8_t value = 0, error = 0;
        int rc = packet->read1ByteTxRx(port.get(), id, addr, &value, &error);
        check(rc, error, id); return value;
    }
    uint32_t read32(int id, uint16_t addr) {
        uint32_t value = 0; uint8_t error = 0;
        int rc = packet->read4ByteTxRx(port.get(), id, addr, &value, &error);
        check(rc, error, id); return value;
    }
    uint16_t read16(int id, uint16_t addr) {
        uint16_t value = 0; uint8_t error = 0;
        int rc = packet->read2ByteTxRx(port.get(), id, addr, &value, &error);
        check(rc, error, id); return value;
    }
    void write16(int id, uint16_t addr, uint16_t value) {
        uint8_t error = 0;
        int rc = packet->write2ByteTxRx(port.get(), id, addr, value, &error);
        check(rc, error, id);
        if (read16(id, addr) != value) throw std::runtime_error("gain readback failed");
    }
    void prepare_teach() {
        for (int id = 0; id < 2; ++id) {
            original_p[id] = read16(id, 84); original_d[id] = read16(id, 80);
            if (!original_p[id] || read16(id, 82) != 0)
                throw std::runtime_error("teach requires a nonzero holding P gain and zero I gain");
        }
        for (int id = 0; id < 2; ++id) {
            changed_gains[id] = true;
            // Vendor manual-placement mode, using correct 2-byte gain writes.
            write16(id, 84, 0); write16(id, 80, 100);
        }
    }
    void restore_gains(int id) {
        if (changed_gains[id]) {
            write16(id, 80, original_d[id]); write16(id, 84, original_p[id]);
            changed_gains[id] = false;
        }
    }
    void write8(int id, uint16_t addr, uint8_t value) {
        uint8_t error = 0;
        int rc = packet->write1ByteTxRx(port.get(), id, addr, value, &error);
        check(rc, error, id);
    }
    void write32(int id, uint16_t addr, uint32_t value) {
        uint8_t error = 0;
        int rc = packet->write4ByteTxRx(port.get(), id, addr, value, &error);
        check(rc, error, id);
    }
    void inspect(int id) {
        uint16_t model = 0; uint8_t error = 0;
        int rc = packet->ping(port.get(), id, &model, &error); check(rc, error, id);
        if (model != 1060 && model != 1240) throw std::runtime_error("unsupported servo model " + std::to_string(model));
        uint8_t m = read8(id, mode), t = read8(id, torque), e = read8(id, hardware_error);
        uint32_t p = read32(id, present);
        std::cout << "servo=" << id << " model=" << model << " mode=" << int(m)
                  << " torque=" << int(t) << " position=" << p << " hardware_error=" << int(e) << std::endl;
        if (e) throw std::runtime_error("servo hardware error");
    }
    uint32_t prepare(int id) {
        if (read8(id, mode) != 3) throw std::runtime_error("hold requires existing position mode 3; no automatic mode change");
        if (read8(id, torque) != 0) throw std::runtime_error("servo already enabled; stop its existing controller first");
        auto p = read32(id, present);
        if (p > 4095) throw std::runtime_error("position outside single-turn range; calibration/mode check required");
        return p;
    }
    void validate_target(int id, uint32_t current, double destination) {
        auto minimum = read32(id, 52), maximum = read32(id, 48);
        if (minimum > maximum || maximum > 4095 || current < minimum || current > maximum ||
            destination < minimum || destination > maximum)
            throw std::runtime_error("pose outside servo position limits; check calibration (no wraparound permitted)");
    }
    void feedback(int id, uint32_t target) {
        if (read8(id, hardware_error) || read8(id, torque) != 1 ||
            std::abs(int64_t(read32(id, present)) - int64_t(target)) > 32)
            throw std::runtime_error("head position/torque feedback check failed");
    }
    void enable(int id, uint32_t p) {
        // Set the current position as the target before enabling torque.
        write32(id, goal, p);
        enabled[id] = true;  // Even a failed enable acknowledgement needs cleanup.
        write8(id, torque, 1);
        if (read8(id, torque) != 1) throw std::runtime_error("torque enable readback failed");
    }
    bool disable() noexcept {
        bool ok = true;
        for (int id = 0; id < 2; ++id) if (enabled[id]) {
            try {
                write8(id, torque, 0);
                if (read8(id, torque) != 0) throw std::runtime_error("torque disable readback failed");
                enabled[id] = false;
            } catch (const std::exception& e) { std::cerr << "DISABLE FAILED: " << e.what() << std::endl; ok = false; }
        }
        for (int id = 0; id < 2; ++id) if (changed_gains[id] && !enabled[id]) {
            try { restore_gains(id); }
            catch (const std::exception& e) { std::cerr << "GAIN RESTORE FAILED: " << e.what() << std::endl; ok = false; }
        }
        return ok;
    }
    ~Bus() {
        disable();
        if (port) port->closePort();
        if (lock_fd >= 0) { ioctl(lock_fd, TIOCNXCL); close(lock_fd); }
    }
};
}
int main(int argc, char** argv) {
    try {
        auto o = parse(argc, argv);
        std::signal(SIGINT, stop); std::signal(SIGTERM, stop); std::signal(SIGHUP, stop);
        Bus bus(o.device);
        for (int id = 0; id < 2; ++id) bus.inspect(id);
        std::array<uint32_t, 2> target{};
        if (!o.probe) {
            // Validate BOTH axes before enabling either one.
            for (int id = 0; id < 2; ++id) target[id] = bus.prepare(id);
            auto initial = target;
            if (o.raw) for (int id = 0; id < 2; ++id) {
                target[id] = uint32_t(o.encoder[id]);
                bus.validate_target(id, initial[id], target[id]);
                std::cout << "joint=" << id << " raw_target_encoder=" << target[id] << std::endl;
            }
            if (o.target_pose) for (int id = 0; id < 2; ++id) {
                // Same lower-limit-relative conversion as Unitree utilities::angle2encoder.
                double value = o.calibration[id] + direction[id] * (o.degrees[id] - lower[id]) * counts_per_degree;
                if (value < 0 || value > 4095) throw std::runtime_error("calibrated target outside single-turn range");
                double angle = lower[id] + direction[id] * (double(initial[id]) - o.calibration[id]) / counts_per_degree;
                if (angle < lower[id] - 1 / counts_per_degree || angle > upper[id] + 1 / counts_per_degree)
                    throw std::runtime_error("startup pose outside calibrated joint limits; check calibration");
                target[id] = uint32_t(std::lround(value));
                bus.validate_target(id, initial[id], target[id]);
                std::cout << "joint=" << id << " target_deg=" << o.degrees[id] << " target_encoder=" << target[id] << std::endl;
            }
            if (o.teach) bus.prepare_teach();
            for (int id = 0; id < 2 && !stopped; ++id) bus.enable(id, initial[id]);
            if (o.teach && !stopped) {
                std::cout << "TEACH READY: torque ON with damping; support and position the head. Send hold to capture and lock without torque-off; stop to release. Timeout 300 seconds." << std::endl;
                auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
                bool captured = false;
                std::string input;
                while (!stopped && !captured) {
                    if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("teach timeout; releasing torque");
                    for (int id = 0; id < 2; ++id)
                        if (bus.read8(id, hardware_error) || bus.read8(id, torque) != 1)
                            throw std::runtime_error("teach hardware/torque fault");
                    fd_set fds; FD_ZERO(&fds); FD_SET(STDIN_FILENO, &fds);
                    timeval tv{0, 100000};
                    if (select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) <= 0) continue;
                    char c;
                    if (read(STDIN_FILENO, &c, 1) != 1) throw std::runtime_error("teach input closed");
                    if (c != '\n') { if (c != '\r') input += c; continue; }
                    if (input == "stop") { stopped = 1; break; }
                    if (input != "hold") { input.clear(); std::cout << "Send hold or stop" << std::endl; continue; }
                    input.clear();
                    std::array<uint32_t, 2> lo{}, hi{};
                    for (int id = 0; id < 2; ++id) lo[id] = hi[id] = bus.read32(id, present);
                    for (int n = 0; n < 4 && !stopped; ++n) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        for (int id = 0; id < 2; ++id) {
                            target[id] = bus.read32(id, present);
                            lo[id] = std::min(lo[id], target[id]); hi[id] = std::max(hi[id], target[id]);
                        }
                    }
                    if (stopped) break;
                    if (hi[0]-lo[0] > 4 || hi[1]-lo[1] > 4) {
                        std::cout << "POSE MOVING: support steadily, then send hold again; torque remains ON" << std::endl; continue;
                    }
                    for (int id = 0; id < 2; ++id) {
                        bus.validate_target(id, target[id], target[id]);
                        bus.write32(id, goal, target[id]);
                    }
                    for (int id = 0; id < 2; ++id) bus.restore_gains(id);
                    for (int id = 0; id < 2; ++id) bus.feedback(id, target[id]);
                    initial = target;
                    captured = true;
                    std::cout << "CAPTURED: joint0_encoder=" << target[0] << " joint1_encoder=" << target[1] << "; torque stayed ON" << std::endl;
                }
            }
            // Small setpoint increments; never jump straight from startup to the target.
            auto command = initial;
            constexpr double step = 10 * counts_per_degree * .02;
            double travel = std::max(std::abs(double(target[0]) - initial[0]), std::abs(double(target[1]) - initial[1]));
            int steps = int(std::ceil(travel / step));
            for (int n = 1; n <= steps && !stopped; ++n) {
                for (int id = 0; id < 2 && !stopped; ++id) {
                    bus.feedback(id, command[id]);
                    command[id] = uint32_t(std::lround(initial[id] + (double(target[id]) - initial[id]) * n / steps));
                    bus.write32(id, goal, command[id]);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (!stopped) {
                for (int id = 0; id < 2; ++id) bus.feedback(id, target[id]);
                std::cout << "READY: both head servos holding pose" << std::endl;
            }
        }
        auto start = std::chrono::steady_clock::now();
        while (!stopped) {
            for (int id = 0; id < 2; ++id) {
                if (bus.read8(id, hardware_error)) throw std::runtime_error("servo hardware error during monitoring");
                auto p = bus.read32(id, present);
                if (!o.probe && (bus.read8(id, torque) != 1 ||
                    std::abs(int64_t(p) - int64_t(target[id])) > 32))
                    throw std::runtime_error("hold feedback/torque check failed");
            }
            if (o.duration > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() >= o.duration) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!bus.disable()) return 2;
        if (stopped) { std::cout << "STOPPED: cleanup complete" << std::endl; return 130; }
        std::cout << (o.probe ? "PASS: both head servos replied; no control registers written" : "PASS: bounded hold completed; torque disabled") << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
