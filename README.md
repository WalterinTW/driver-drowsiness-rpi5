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
