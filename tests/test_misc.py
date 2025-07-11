from unittest.mock import MagicMock, patch

from napari.utils.notifications import Notification, NotificationSeverity, notification_manager

from cell_gater.utils.misc import napari_notification


def test_napari_notification():
    with patch("cell_gater.utils.misc.notification_manager.dispatch") as mock_dispatch:
        with patch("cell_gater.utils.misc.Notification") as mock_notification_class:
            test_message = "Test notification"
            napari_notification(test_message)

            mock_notification_class.assert_called_once_with(test_message, severity=NotificationSeverity.INFO)
            mock_dispatch.assert_called_once_with(mock_notification_class.return_value)

    with patch("cell_gater.utils.misc.notification_manager.dispatch") as mock_dispatch:
        with patch("cell_gater.utils.misc.Notification") as mock_notification_class:
            test_message = "Warning notification"
            napari_notification(test_message, severity=NotificationSeverity.WARNING)

            mock_notification_class.assert_called_once_with(test_message, severity=NotificationSeverity.WARNING)
            mock_dispatch.assert_called_once_with(mock_notification_class.return_value)
