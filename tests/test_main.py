from driver_drowsiness.main import main
from driver_drowsiness.main import build_parser


def test_main_without_demo_prints_startup_message(capsys):
    main([])

    captured = capsys.readouterr()
    assert "Driver drowsiness detection app starting" in captured.out


def test_parser_accepts_step_7_source_and_backend_options():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--demo",
            "webcam",
            "--source",
            "rpi-camera",
            "--backend",
            "hailo",
        ]
    )

    assert args.source == "rpi-camera"
    assert args.backend == "hailo"
