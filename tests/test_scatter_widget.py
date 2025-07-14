from pathlib import Path
from napari.layers import Image, Points
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from qtpy import QtCore
from qtpy.QtWidgets import QComboBox, QLineEdit, QPushButton

from cell_gater.model.data_model import DataModel
from cell_gater.utils.csv_df import stack_csv_files
from cell_gater.widgets.scatter_widget import ScatterInputWidget


@pytest.fixture
def mock_viewer():
    viewer = MagicMock()
    viewer.window = MagicMock()
    viewer.window.window_menu = MagicMock()
    viewer.layers = MagicMock()
    viewer.layers.select_all = MagicMock()
    viewer.layers.remove_selected = MagicMock()
    viewer.add_image = MagicMock()
    viewer.add_labels = MagicMock()
    viewer.add_points = MagicMock()
    return viewer

@pytest.fixture
def populated_data_model():
    model = DataModel()
    test_quants_path = Path(__file__).parent / "test_data" / "quants"
    test_imgs_path = Path(__file__).parent / "test_data" / "imgs"
    test_segs_path = Path(__file__).parent / "test_data" / "segs"

    model.regionprops_df = stack_csv_files(test_quants_path)
    model.image_paths = [p for p in test_imgs_path.iterdir() if not p.name.startswith('.')]
    model.mask_paths = [p for p in test_segs_path.iterdir() if not p.name.startswith('.')]
    image_stems = {i.stem.replace('.ome', '') for i in model.image_paths}
    mask_stems = {i.stem.replace('.ome', '') for i in model.mask_paths}
    assert image_stems == mask_stems
    model.samples = sorted(list(image_stems))
    model.sample_image_mapping = {p.stem.replace('.ome', ''): p for p in model.image_paths}
    model.sample_mask_mapping = {p.stem.replace('.ome', ''): p for p in model.mask_paths}
    model.lower_bound_marker = model.regionprops_df.columns[1]
    model.upper_bound_marker = model.regionprops_df.columns[-1]
    model.markers = [col for col in model.regionprops_df.columns if col.startswith("mean_")] # Example markers
    model.markers_image_indices = {marker: i for i, marker in enumerate(model.markers)} # Dummy indices
    model.active_sample = model.samples[0]
    model.active_marker = model.markers[0]
    model.active_y_axis = "Area"
    model.active_ref_marker = model.markers[0]
    model.validated = True
    return model

def test_scatter_widget_initialization(mock_viewer, populated_data_model, qtbot):
    with patch("cell_gater.widgets.scatter_widget.imread", return_value=np.zeros((10, 10))) as mock_imread:
        widget = ScatterInputWidget(populated_data_model, mock_viewer)
        qtbot.addWidget(widget)

        assert widget._model == populated_data_model
        assert widget._viewer == mock_viewer
        assert isinstance(widget.sample_selection_dropdown, QComboBox)
        assert isinstance(widget.marker_selection_dropdown, QComboBox)
        assert isinstance(widget.choose_y_axis_dropdown, QComboBox)
        assert isinstance(widget.ref_channel_dropdown, QComboBox)
        assert isinstance(widget.log_scale_dropdown, QComboBox)
        assert isinstance(widget.plot_type_dropdown, QComboBox)
        assert isinstance(widget.manual_gate_input_text, QLineEdit)
        assert isinstance(widget.manual_gate_input_QPushButton, QPushButton)

        # Check dropdown populations
        assert widget.sample_selection_dropdown.count() == len(populated_data_model.samples)
        assert widget.marker_selection_dropdown.count() == len(populated_data_model.markers)
        assert widget.choose_y_axis_dropdown.count() == len(populated_data_model.regionprops_df.columns)

        # Check initial active selections
        assert widget.sample_selection_dropdown.currentText() == populated_data_model.active_sample
        assert widget.marker_selection_dropdown.currentText() == populated_data_model.active_marker
        assert widget.choose_y_axis_dropdown.currentText() == populated_data_model.active_y_axis
        assert widget.ref_channel_dropdown.currentText() == populated_data_model.active_ref_marker

        # Verify initial data loading calls
        mock_imread.assert_called()
        mock_viewer.add_image.assert_called()
        mock_viewer.add_labels.assert_called()

def test_scatter_widget_manual_gate_input(mock_viewer, populated_data_model, qtbot):
    with patch("cell_gater.widgets.scatter_widget.imread", return_value=np.zeros((10, 10))):
        widget = ScatterInputWidget(populated_data_model, mock_viewer)
        qtbot.addWidget(widget)

        # Simulate typing a value into the manual gate input text field
        test_gate_value = "123.45"
        widget.manual_gate_input_text.setText(test_gate_value)
        
        # Simulate clicking the manual gate input button
        qtbot.mouseClick(widget.manual_gate_input_QPushButton, QtCore.Qt.LeftButton)

        # Assert that the model's current_gate is updated correctly
        assert populated_data_model.current_gate == float(test_gate_value)


