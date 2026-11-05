# Revo3 Live Mirror

A desktop URDF viewer that follows the measured joint angles of a Revo3 hand
over RS485. Includes left and right visual models and a zero-force / pose-hold
switch for all 21 joints.

## Install

Requires Python 3.10 or 3.11 with `venv` support, a graphical desktop with OpenGL,
and access to the RS485 serial device. Internet access is needed for the initial
Python dependency installation; normal operation is offline.

```bash
./setup.sh
./run.sh
```

To choose a Python interpreter for installation:

```bash
REVO3_PYTHON=/path/to/python ./setup.sh
```

The installer creates `.venv/` inside this folder. The SDK is pinned to 1.1.7,
the API version used by this viewer. No ROS, PyBullet, or other project checkout
is required.

## Connect

For the right hand used during development:

```bash
./run.sh --connect --port /dev/ttyUSB0 --slave-id 127 --force-side right
```

For the left hand used during development:

```bash
./run.sh --connect --port /dev/ttyUSB0 --slave-id 126 --force-side left
```

Use the port and device ID of your actual hand. Default baud rate is 5 Mbps;
change it with `--baudrate`. Feedback polling defaults to 30 Hz (`--rate 30`).
Close other applications using the same serial port before connecting.

The viewer normally selects the model from device-reported hand type.
`--force-side` overrides that selection when firmware reports the wrong side;
it does not modify device configuration. One window supports one hand. Two
instances can use separate serial ports.

## Use

- **Zero-force ON** is applied on every connection: all joints receive a 0 mA
  current-control target. Move the fingers by hand to inspect alignment.
- Switch **Zero-force OFF** to engage normal position holding. A fresh hardware
  reading supplies all 21 hold targets, so the hand holds the pose you moved it
  to. Switch ON to release it again. Disconnect/close leaves the last commanded
  mode in place; reconnect starts in zero-force.
- Drag to orbit, scroll to zoom, Ctrl-drag to pan, or select a preset view.
- Select a joint row to show its URDF axis in yellow. The table displays raw
  measured degrees without smoothing or clipping. `LIM` marks values outside
  URDF limits; `E:xxxx` preserves raw SDK error bits.
- **Pin reference** keeps a reference skeleton and shows angle differences.
- **Freeze view** pauses the display while acquisition continues.
- **Save snapshot** writes PNG and JSON files into `captures/`. Captures include
  feedback, timestamps, device identity, joint mapping, model hash, and control
  state. The UI reports acknowledged mode commands, not independent firmware
  mode readback.
- Sign/offset previews change only the displayed model. Hold commands always
  use fresh raw device angles, including when the display is frozen.

Offline demo (never opens a hardware connection):

```bash
./run.sh --demo
```

## Package contents

Three Python modules, setup/launch scripts, requirements, and visual assets for
both hands. Virtual environments, captures, tests, and analysis outputs are
not included. `live_model.py` contains the complete SDK-to-URDF motor mapping.

Models derive from BrainCo's `20261014` release, commit
`6db7e2ed1d864ee0a3ab6ff62d43a303787137f6`. Collision elements and collision-only
meshes are omitted because this viewer does not simulate contacts. Joint
definitions, visual geometry, mesh scale, and visual origins are preserved.
These packaged URDFs are for visualization, not collision simulation.

See `MODEL_SOURCE.json` for provenance and asset checksums, and
`UPSTREAM_README.md` for the upstream description and license information.
