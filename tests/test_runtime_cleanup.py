"""Startup failures must close the controllers already created by the launcher."""

from unittest.mock import Mock, patch

from airdesk.virtual_cursor import run_virtual_cursor


def test_missing_model_closes_every_controller():
    resources = [Mock(), Mock(), Mock()]
    with patch("airdesk.virtual_cursor.MODEL_PATH") as model:
        model.exists.return_value = False
        assert run_virtual_cursor(
            system_mouse=resources[0], mode_controller=resources[1], control_panel=resources[2]
        ) == 1
    for resource in resources:
        resource.close.assert_called_once()


def test_camera_denial_releases_camera_and_controllers():
    resources = [Mock(), Mock(), Mock()]
    with patch("airdesk.virtual_cursor.MODEL_PATH") as model, patch(
        "airdesk.virtual_cursor.cv2.VideoCapture"
    ) as camera:
        model.exists.return_value = True
        camera.return_value.isOpened.return_value = False
        assert run_virtual_cursor(
            system_mouse=resources[0], mode_controller=resources[1], control_panel=resources[2]
        ) == 1
        camera.return_value.release.assert_called_once()
    for resource in resources:
        resource.close.assert_called_once()
