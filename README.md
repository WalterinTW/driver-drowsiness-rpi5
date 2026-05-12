# Driver Drowsiness Detection on Raspberry Pi 5

This repository is being developed in small stages for a Raspberry Pi 5
driver drowsiness detection system.

## Staged Development Plan

1. Create the initial Python package skeleton, placeholder configuration, and
   editable install metadata.
2. Add mock camera input so development can continue on Windows and macOS
   before Raspberry Pi hardware is connected.
3. Add feature extraction interfaces for EAR, MAR, and PERCLOS without binding
   them to a specific model runtime.
4. Add decision logic for alert levels using configurable thresholds.
5. Add alert output modules with mock implementations first, then Raspberry Pi
   hardware-backed implementations.
6. Integrate real camera, inference, GPIO, audio, or accelerator support only
   after the module boundaries and mock flow are stable.

## Raspberry Pi 5 Deployment Preparation

The Raspberry Pi 5 and Hailo AI Kit are assumed to be already configured and
validated. This project does not duplicate the Hailo installation flow here.
The goal of this stage is to keep the app movable to the Raspberry Pi while
preserving the Windows 11 MediaPipe webcam demo.

Recommended smoke-test setup on Raspberry Pi 5:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-rpi.txt
python3 -m pip install -e .
python3 -m pytest
```

The current command-line shape is:

```bash
python3 -m driver_drowsiness.main --demo webcam --source webcam --backend mock
python3 -m driver_drowsiness.main --demo webcam --source video --video-path sample.mp4 --backend mock
python3 -m driver_drowsiness.main --demo webcam --source rpi-camera --backend hailo
```

`--source rpi-camera` and `--backend hailo` are deployment placeholders right
now. They are intentionally present so the Raspberry Pi integration can happen
behind the same interfaces as the Windows MediaPipe demo.

The existing working Hailo object detection example should be adapted later in
`src/driver_drowsiness/inference/hailo_landmark.py`. The needed details are:

- Hailo model file path and format for face landmarks.
- Input tensor shape, color format, normalization, and resize/crop behavior.
- Output tensor names, shapes, landmark ordering, and coordinate scaling.
- The exact Hailo runtime API calls used in the working object detection demo.
- Any camera frame format assumptions from the Raspberry Pi pipeline.
