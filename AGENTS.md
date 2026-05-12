# AGENTS.md

## Project target

This project is developed on Windows 11 / macOS, but the final deployment target is:

- Raspberry Pi 5
- 64-bit Raspberry Pi OS
- Python 3.x
- Hardware may include GPIO, camera, I2C, SPI, UART, sensors, motors, or audio/video output.

## Development rules

1. Do not assume the code only runs on Windows or macOS.
2. Avoid Windows-only paths, shell commands, or packages unless clearly isolated.
3. Use `pathlib.Path` for file paths.
4. Keep hardware-dependent code isolated in separate modules.
5. Provide a mock or simulation mode for development on Windows/macOS.
6. Provide Raspberry Pi installation commands when adding new dependencies.
7. When hardware access is required, clearly mark the test as “must be tested on Raspberry Pi 5”.
8. Do not perform large refactoring unless explicitly requested.
9. Preserve existing working features unless the task requires changing them.
10. After each change, explain:
    - modified files
    - how to test on Windows/macOS
    - how to test on Raspberry Pi 5