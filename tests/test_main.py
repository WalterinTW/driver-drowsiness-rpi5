from driver_drowsiness.main import main


def test_main_without_demo_prints_startup_message(capsys):
    main([])

    captured = capsys.readouterr()
    assert "Driver drowsiness detection app starting" in captured.out
