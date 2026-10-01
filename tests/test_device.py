from unittest.mock import MagicMock, patch


def test_auto_device_uses_cuda_when_available(tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"placeholder")
    fake_model = MagicMock()
    with patch("torch.cuda.is_available", return_value=True), \
         patch("ultralytics.YOLO", return_value=fake_model):
        from offtheball.detection import PersonDetector
        detector = PersonDetector(model=model_path, device="auto")
    assert detector.device == 0


def test_explicit_cpu_is_preserved(tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"placeholder")
    with patch("ultralytics.YOLO", return_value=MagicMock()):
        from offtheball.detection import PersonDetector
        detector = PersonDetector(model=model_path, device="cpu", cpu_threads=1)
    assert detector.device == "cpu"
