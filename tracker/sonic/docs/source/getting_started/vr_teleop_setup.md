# VR Teleop Setup (PICO)

This page covers the one-time hardware and software setup for PICO VR whole-body teleoperation. After completing these steps, proceed to the [ZMQ Manager tutorial](../tutorials/vr_wholebody_teleop.md) to run teleop in sim or on real hardware.

---

## Required Hardware

- [PICO 4 / PICO 4 Pro headset](https://www.picoxr.com/global/products/pico4)
- [2x PICO controllers](https://www.picoxr.com/global/products/pico4)
- [2x PICO motion trackers](https://www.picoxr.com/global/products/pico-motion-tracker) (strapped to ankles)
- A compatible headset Ethernet adapter and a wired connection to the PC (direct or on the same wired LAN). WB-WAM collection defaults to Ethernet; use Wi-Fi only when explicitly selected.

For WB-WAM's complete wiring, actual PICO/PC IP configuration, and PC-connected MANUS USB receiver checks, follow the [collection guide](../../../../../collector/sonic/README.md#1a-wired-pico-setup). The robot also connects to the PC by Ethernet; its camera and Wuji hands remain attached to the robot by USB.

---

## Step 1: Install XRoboToolkit

XRoboToolkit consists of a PC service (running on your workstation) and a PICO app (running on the headset) that streams body-tracking data.

### PC Service

The PC service must be installed and running on your workstation **before** the PICO can connect.

**Ubuntu 22.04 (x86_64 workstation):**

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

**Ubuntu 24.04 (x86_64 workstation):**

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
```

**Jetson (aarch64, onboard):**

```bash
sudo dpkg -i gear_sonic_deploy/thirdparty/roboticsservice_1.0.0.0_arm64.deb
```

See [XRoboToolkit-PC-Service releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases) for other platforms or newer versions.

### PICO App

1. Wear the PICO headset to begin the setup and installation process.
2. Complete the quick setup on PICO.
3. Make sure PICO has Internet access for the download. Wi-Fi may be used temporarily here; use the configured Ethernet link for collection.
4. Open the browser application in the PICO.
5. Type **"xrobotoolkit"** in the search bar and select the GitHub page [https://github.com/XR-Robotics](https://github.com/XR-Robotics).

```{image} ../_static/pico_setup/google_search_screenshot.png
:width: 600px
:align: center
```

6. Make sure **Developer Mode** is enabled (Settings → Developer).
7. **[INSIDE PICO]** Scroll down in the GitHub page until you see the APK download option and click with the PICO trigger to download it.

```{tip}
Download [XRoboToolkit-PICO-1.1.1.apk](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk) on PICO using the browser. ([Other Versions](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases))
```

8. **[INSIDE PICO]** Open the manage downloads option on the top right section of the browser page and click to open the `XRoboToolkit-PICO-1.1.1.apk` download.
9. **[INSIDE PICO]** Select **Install** — the application will appear in the **Unknown** section of your library.

---

## Step 2: Motion Tracker Setup

```{image} ../_static/pico_setup/pico_setup_screenshot.png
:width: 600px
:align: center
```

1. Strap one PICO motion tracker to your left ankle and one to your right ankle. **Scrunch** down any baggy clothing so the trackers are visible. Make sure the side with the light indicator faces up.
2. Go to PICO settings. In the menu on the left, scroll down to the last option: **"Developer"**. Make sure **"Safeguard"** is turned off.
   - If the Developer option is not active, tap on "Software" until it appears.
3. Click the **Wi-Fi icon** in the PICO menu. A picture of the headset will appear. Above the headset, there will be a small circular logo for the motion trackers. If there is no logo, open the **"Motion Tracker"** app itself.
   - Headset and 2 controllers will populate — select **Motion Tracker** (small circle).
4. Next to each tracker, there is an **"i"** icon. Click on this and **unpair all trackers**.
5. Once all trackers are cleared, click the **"Pair"** button in the top right corner.
6. Press and hold the button on the top of each motion tracker for **6 seconds**. Once in pairing mode, the lights will flash red and blue.

### Motion Tracker Calibration

1. Wear the PICO headset over your eyes.
2. Press the blue **"Calibrate"** button and follow the two calibration sequences:
   - **Sequence 1:** Stand stiff with the handheld controllers down by your sides.
   - **Sequence 2:** Look down at the foot motion trackers until the headset cameras recognize them.
3. Once calibrated, wear the PICO headset around your forehead (ensuring PICO faces forward to continue detecting motion trackers).

---

## Step 3: Install the PICO Teleop Environment

From the **repo root**:

```bash
bash install_scripts/install_pico.sh
```

This creates a `.venv_teleop` virtual environment (Python 3.10) that includes:
- `teleop` extra (ZMQ, Pinocchio, PyVista)
- `sim` extra (MuJoCo, tyro)
- XRoboToolkit SDK
- Unitree SDK2 Python bindings

Activate it with:

```bash
source .venv_teleop/bin/activate   # prompt: (gear_sonic_teleop)
```

---

## Step 4: Connect the PICO to Your Workstation

1. Connect PICO and the PC by **Ethernet** and obtain different IPv4 addresses in the same subnet. Use an existing DHCP network, or configure a dedicated PC link as described in the WB-WAM collection guide. Record PICO's wired IP and use `ip route get <PICO_IP>` on the PC to verify the Ethernet interface and PC source IP.
   - Read the headset address from its Ethernet network details / XRoboToolkit Network panel. A displayed Wi-Fi address is not proof of the wired address. The screenshots below show example interfaces and addresses, not values to reuse.

```{image} ../_static/pico_setup/internet.png
:width: 600px
:align: center
```

```{image} ../_static/pico_setup/pico_vr_screenshot.png
:width: 600px
:align: center
```

2. Open the **XRoboToolKit** application. Enter the **PC's PICO-facing Ethernet IP** by clicking **"Enter"** next to "PC Service:". This is not PICO's own IP or the robot IP. You will know it is properly connected if **WORKING** appears next to "Status:".
   - If your IP address is already inputted, select **"Reconnect"** where it says "Status:" in the Network section.

3. Make sure the following boxes are ticked as shown in the picture below:
   - **"Head"** and **"Controller"** under the "Tracking" section.
   - For Data/Control, make sure to select the **"Send"** button.
   - For "Pico Motion Tracker" make sure to select **"Full body"**.

```{image} ../_static/pico_setup/xrrobot_setup.png
:width: 600px
:align: center
```

---

## Next Steps

For WB-WAM, run the bounded PICO and MANUS input checks and configure video return as described in the [collection guide](../../../../../collector/sonic/README.md#1a-wired-pico-setup) before reporting readiness. The [ZMQ Manager (`zmq_manager`) tutorial](../tutorials/vr_wholebody_teleop.md) provides further teleoperation details.
