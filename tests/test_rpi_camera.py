import pytest

from driver_drowsiness.camera.rpi_camera import RaspberryPiCamera


def test_rpi_camera_placeholder_has_frame_source_interface():
    camera = RaspberryPiCamera()

    with pytest.raises(NotImplementedError, match="Raspberry Pi"):
        camera.open()


def test_rpi_camera_placeholder_supports_release():
    camera = RaspberryPiCamera()

    camera.release()
