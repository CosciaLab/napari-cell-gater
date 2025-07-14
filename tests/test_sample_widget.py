from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy
import pandas as pd
import pytest
from qtpy import QtCore
from qtpy.QtWidgets import QComboBox, QLineEdit, QPushButton

from cell_gater.model.data_model import DataModel
from cell_gater.utils.csv_df import stack_csv_files
from cell_gater.widgets.sample_widget import SampleWidget


# Use a MagicMock for napari.Viewer
@pytest.fixture
def mock_viewer():
    viewer = MagicMock()
    viewer.window = MagicMock()
    viewer.window.window_menu = MagicMock()
    return viewer

@pytest.fixture
def sample_df():
    # Use a real quantification CSV for realistic DataFrame
    test_data_dir = Path(__file__).parent / "test_data" / "quants"
    df = pd.read_csv(test_data_dir / "1.csv")
    return df

@pytest.fixture
def data_model():
    model = DataModel()
    # Initialize with empty dataframes/lists to allow tests to populate them
    model.regionprops_df = pd.DataFrame()
    model.image_paths = []
    model.mask_paths = []
    model.samples = []
    model.sample_image_mapping = {}
    model.sample_mask_mapping = {}
    return model

def test_widget_initialization(mock_viewer, data_model, qtbot):
    widget = SampleWidget(mock_viewer, data_model)
    qtbot.addWidget(widget)
    assert widget.viewer == mock_viewer
    assert widget.model == data_model
    assert widget.layout().count() > 0
    assert isinstance(widget.load_samples_button, QPushButton)
    assert isinstance(widget.load_image_dir_button, QPushButton)
    assert isinstance(widget.load_mask_dir_button, QPushButton)
    assert isinstance(widget.lower_bound_marker_col, QComboBox)
    assert isinstance(widget.upper_bound_marker_col, QComboBox)
    assert isinstance(widget.filter_field, QLineEdit)
    assert isinstance(widget.validate_button, QPushButton)

def test_set_image_paths(mock_viewer, data_model):
    widget = SampleWidget(mock_viewer, data_model)
    with patch("cell_gater.widgets.sample_widget.napari_notification") as mock_notify:
        widget._set_image_paths("/tmp")
        mock_notify.assert_called()
        assert isinstance(widget.model.image_paths, list)

def test_set_mask_paths(mock_viewer, data_model):
    widget = SampleWidget(mock_viewer, data_model)
    with patch("cell_gater.widgets.sample_widget.napari_notification") as mock_notify:
        widget._set_mask_paths("/tmp")
        mock_notify.assert_called()
        assert isinstance(widget.model.mask_paths, list)

def test_assign_regionprops_to_model(mock_viewer, data_model):
    widget = SampleWidget(mock_viewer, data_model)
    with patch("cell_gater.widgets.sample_widget.stack_csv_files", return_value=data_model.regionprops_df) as mock_stack:
        widget._assign_regionprops_to_model("/tmp")
        mock_stack.assert_called_with(Path("/tmp"))
        assert isinstance(widget.model.regionprops_df, pd.DataFrame)

def test_update_model_lowerbound_and_upperbound(mock_viewer, data_model):
    widget = SampleWidget(mock_viewer, data_model)
    widget.lower_bound_marker_col.addItem("CD3")
    widget.upper_bound_marker_col.addItem("CD8")
    widget.lower_bound_marker_col.setCurrentText("CD3")
    widget.upper_bound_marker_col.setCurrentText("CD8")
    widget._update_model_lowerbound()
    widget._update_model_upperbound()
    assert widget.model.lower_bound_marker == "CD3"
    assert widget.model.upper_bound_marker == "CD8"

def test_update_filter(mock_viewer, data_model):
    widget = SampleWidget(mock_viewer, data_model)
    widget.filter_field.setText("CD3,CD8")
    widget._update_filter()
    assert widget.model.marker_filter == "CD3,CD8"

def test_validate_success(mock_viewer, data_model):
    widget = SampleWidget(mock_viewer, data_model)

    # Set up the model with valid data for validation
    test_quants_path = Path(__file__).parent / "test_data" / "quants"
    test_imgs_path = Path(__file__).parent / "test_data" / "imgs"
    test_segs_path = Path(__file__).parent / "test_data" / "segs"

    data_model.regionprops_df = stack_csv_files(test_quants_path)
    data_model.image_paths = [p for p in test_imgs_path.iterdir() if not p.name.startswith('.')]
    data_model.mask_paths = [p for p in test_segs_path.iterdir() if not p.name.startswith('.')]
    data_model.lower_bound_marker = data_model.regionprops_df.columns[1]
    data_model.upper_bound_marker = data_model.regionprops_df.columns[-1]
    data_model.marker_filter = ""

    with patch("cell_gater.widgets.sample_widget.napari_notification") as mock_notify, patch(
        "cell_gater.widgets.sample_widget.dask_image.imread.imread", return_value=numpy.array([[0]])
    ) as mock_imread, patch("cell_gater.widgets.sample_widget.ScatterInputWidget") as mock_scatter:
        widget._validate()
        mock_notify.assert_called()
        assert widget.model.validated is True

def test_validate_failure_missing_data(mock_viewer, data_model):
    widget = SampleWidget(mock_viewer, data_model)
    widget.model.regionprops_df = pd.DataFrame()
    with pytest.raises(AssertionError):
        widget._validate()
    widget.model.regionprops_df = data_model.regionprops_df
    widget.model.image_paths = []
    with pytest.raises(AssertionError):
        widget._validate()
    widget.model.image_paths = data_model.image_paths
    widget.model.mask_paths = []
    with pytest.raises(AssertionError):
        widget._validate()
    widget.model.mask_paths = data_model.mask_paths
    widget.model.image_paths = [Path("/tmp/img1.tif")]
    widget.model.mask_paths = [Path("/tmp/mask1.tif"), Path("/tmp/mask2.tif")]
    with pytest.raises(AssertionError):
        widget._validate()

def test_open_sample_dialog(mock_viewer, data_model, qtbot):
    widget = SampleWidget(mock_viewer, data_model)
    
    # Mock the file dialog to return our test data path
    test_quants_path = Path(__file__).parent / "test_data" / "quants"
    with patch("cell_gater.widgets.sample_widget.QFileDialog.getExistingDirectory", return_value=str(test_quants_path)):
        # Simulate clicking the load_samples_button
        qtbot.mouseClick(widget.load_samples_button, QtCore.Qt.LeftButton)

        # Assert that the model's regionprops_df is updated
        assert not widget.model.regionprops_df.empty

def test_open_image_dialog(mock_viewer, data_model, qtbot):
    widget = SampleWidget(mock_viewer, data_model)
    
    # Mock the file dialog to return our test data path
    test_imgs_path = Path(__file__).parent / "test_data" / "imgs"
    with patch("cell_gater.widgets.sample_widget.QFileDialog.getExistingDirectory", return_value=str(test_imgs_path)):
        # Simulate clicking the load_image_dir_button
        qtbot.mouseClick(widget.load_image_dir_button, QtCore.Qt.LeftButton)

        # Assert that the model's image_paths are updated
        assert len(widget.model.image_paths) == 5
        assert all(isinstance(p, Path) for p in widget.model.image_paths)

def test_open_mask_dialog(mock_viewer, data_model, qtbot):
    widget = SampleWidget(mock_viewer, data_model)
    
    # Mock the file dialog to return our test data path
    test_segs_path = Path(__file__).parent / "test_data" / "segs"
    with patch("cell_gater.widgets.sample_widget.QFileDialog.getExistingDirectory", return_value=str(test_segs_path)):
        # Simulate clicking the load_mask_dir_button
        qtbot.mouseClick(widget.load_mask_dir_button, QtCore.Qt.LeftButton)

        # Assert that the model's mask_paths are updated
        assert len(widget.model.mask_paths) == 5
        assert all(isinstance(p, Path) for p in widget.model.mask_paths)

def test_update_model_lowerbound_and_upperbound_via_ui(mock_viewer, data_model, qtbot):
    widget = SampleWidget(mock_viewer, data_model)
    
    # Populate the dropdowns with some dummy markers
    widget.lower_bound_marker_col.addItem("MarkerA")
    widget.lower_bound_marker_col.addItem("MarkerB")
    widget.upper_bound_marker_col.addItem("MarkerA")
    widget.upper_bound_marker_col.addItem("MarkerB")

    # Simulate selecting "MarkerA" for lower bound
    widget.lower_bound_marker_col.setCurrentIndex(0) # Select "MarkerA"
    widget.lower_bound_marker_col.currentTextChanged.emit("MarkerA") # Explicitly emit signal
    assert widget.model.lower_bound_marker == "MarkerA"

    # Simulate selecting "MarkerB" for upper bound
    widget.upper_bound_marker_col.setCurrentIndex(1) # Select "MarkerB"
    widget.upper_bound_marker_col.currentTextChanged.emit("MarkerB") # Explicitly emit signal
    assert widget.model.upper_bound_marker == "MarkerB"

def test_update_filter_via_ui(mock_viewer, data_model, qtbot):
    widget = SampleWidget(mock_viewer, data_model)
    
    widget.filter_field.clear()
    # Simulate typing into the filter field
    qtbot.keyClicks(widget.filter_field, "CD3,CD8")
    widget.filter_field.editingFinished.emit() # Explicitly emit signal
    assert widget.model.marker_filter == "CD3,CD8"

def test_validate_button_click(mock_viewer, data_model, qtbot):
    widget = SampleWidget(mock_viewer, data_model)

    # Set up the model with valid data for validation
    test_quants_path = Path(__file__).parent / "test_data" / "quants"
    test_imgs_path = Path(__file__).parent / "test_data" / "imgs"
    test_segs_path = Path(__file__).parent / "test_data" / "segs"

    data_model.regionprops_df = stack_csv_files(test_quants_path)
    data_model.image_paths = [p for p in test_imgs_path.iterdir() if not p.name.startswith('.')]
    data_model.mask_paths = [p for p in test_segs_path.iterdir() if not p.name.startswith('.')]
    data_model.lower_bound_marker = data_model.regionprops_df.columns[1]
    data_model.upper_bound_marker = data_model.regionprops_df.columns[-1]
    data_model.marker_filter = ""

    with patch("cell_gater.widgets.sample_widget.napari_notification") as mock_notify, patch(
        "cell_gater.widgets.sample_widget.dask_image.imread.imread", return_value=numpy.array([[0]])
    ) as mock_imread, patch("cell_gater.widgets.sample_widget.ScatterInputWidget") as mock_scatter_widget_init:
        # Simulate clicking the validate button
        qtbot.mouseClick(widget.validate_button, QtCore.Qt.LeftButton)

        # Assertions
        mock_notify.assert_called_once()
        assert data_model.validated is True
        mock_scatter_widget_init.assert_called_once()
