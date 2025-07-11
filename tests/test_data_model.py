from pathlib import Path

import pandas as pd
import pytest

from cell_gater.model.data_model import DataModel

TEST_DATA_DIR = Path(__file__).parent / "test_data"
IMG_DIR = TEST_DATA_DIR / "imgs"
SEGS_DIR = TEST_DATA_DIR / "segs"
QUANTS_DIR = TEST_DATA_DIR / "quants"

@pytest.fixture
def sample_df():
    # Use one of the quantification CSVs for realistic DataFrame
    df = pd.read_csv(QUANTS_DIR / "1.csv")
    return df

@pytest.fixture
def data_model():
    return DataModel()

def test_initialization(data_model):
    assert isinstance(data_model.events, object)
    assert data_model.samples == []
    assert isinstance(data_model.regionprops_df, pd.DataFrame)
    assert data_model.image_paths == []
    assert data_model.mask_paths == []
    assert data_model.markers == []
    assert data_model.marker_filter == "dna,dapi"
    assert not data_model.validated
    assert data_model.plot_type == "scatter"
    assert isinstance(data_model.gates, pd.DataFrame)
    assert data_model.current_gate == 0.0

def test_setters_and_getters(data_model, sample_df):
    # samples
    samples = ["sample1", "sample2"]
    data_model.samples = samples
    assert data_model.samples == samples

    # regionprops_df
    called = {}
    def on_regionprops_df():
        called['regionprops'] = True
    data_model.events.regionprops_df.connect(on_regionprops_df)
    data_model.regionprops_df = sample_df
    assert data_model.regionprops_df.equals(sample_df)
    assert called.get('regionprops')

    # image_paths
    img_paths = [IMG_DIR / "1.tif", IMG_DIR / "2.tif"]
    data_model.image_paths = img_paths
    assert data_model.image_paths == img_paths

    # mask_paths
    mask_paths = [SEGS_DIR / "1.tif", SEGS_DIR / "2.tif"]
    data_model.mask_paths = mask_paths
    assert data_model.mask_paths == mask_paths

    # sample_image_mapping
    mapping = {"sample1": IMG_DIR / "1.tif"}
    data_model.sample_image_mapping = mapping
    assert data_model.sample_image_mapping == mapping

    # sample_mask_mapping
    mask_mapping = {"sample1": SEGS_DIR / "1.tif"}
    data_model.sample_mask_mapping = mask_mapping
    assert data_model.sample_mask_mapping == mask_mapping

    # lower/upper bound marker
    data_model.lower_bound_marker = "CD3"
    assert data_model.lower_bound_marker == "CD3"
    data_model.upper_bound_marker = "CD8"
    assert data_model.upper_bound_marker == "CD8"

    # markers and indices
    markers = ["CD3", "CD8"]
    data_model.markers = markers
    assert data_model.markers == markers
    indices = ["0", "1"]
    data_model.markers_image_indices = indices
    assert data_model.markers_image_indices == indices

    # active_marker, active_sample, active_y_axis, active_ref_marker
    data_model.active_marker = "CD3"
    assert data_model.active_marker == "CD3"
    data_model.active_sample = "sample1"
    assert data_model.active_sample == "sample1"
    data_model.active_y_axis = "CD8"
    assert data_model.active_y_axis == "CD8"
    data_model.active_ref_marker = "CD4"
    assert data_model.active_ref_marker == "CD4"

    # log_scale and plot_type
    data_model.log_scale = True
    assert data_model.log_scale is True
    data_model.plot_type = "histogram"
    assert data_model.plot_type == "histogram"

    # marker_filter
    data_model.marker_filter = "CD3,CD8"
    assert data_model.marker_filter == "CD3,CD8"

    # validated and event
    called = {}
    def on_validated():
        called['validated'] = True
    data_model.events.validated.connect(on_validated)
    data_model.validated = True
    assert data_model.validated is True
    assert called.get('validated')

    # gates
    gates_df = pd.DataFrame({"gate": [1, 2, 3]})
    data_model.gates = gates_df
    assert data_model.gates.equals(gates_df)

    # current_gate
    data_model.current_gate = 2.5
    assert data_model.current_gate == 2.5

    # manual_channel_mapping
    data_model.manual_channel_mapping = "manual_map"
    assert data_model.manual_channel_mapping == "manual_map"

# Additional integration test using all test data

def test_integration_with_test_data(data_model):
    # Load all images and masks
    img_paths = sorted(IMG_DIR.glob("*.tif"))
    mask_paths = sorted(SEGS_DIR.glob("*.tif"))
    samples = [f"sample{i+1}" for i in range(len(img_paths))]
    data_model.samples = samples
    data_model.image_paths = img_paths
    data_model.mask_paths = mask_paths
    data_model.sample_image_mapping = dict(zip(samples, img_paths, strict=False))
    data_model.sample_mask_mapping = dict(zip(samples, mask_paths, strict=False))
    assert len(data_model.samples) == 5
    assert len(data_model.image_paths) == 5
    assert len(data_model.mask_paths) == 5
    assert set(data_model.sample_image_mapping.keys()) == set(samples)
    assert set(data_model.sample_mask_mapping.keys()) == set(samples)

    # Load and set regionprops_df from all quant CSVs
    dfs = [pd.read_csv(p) for p in sorted(QUANTS_DIR.glob("*.csv"))]
    regionprops_df = pd.concat(dfs, ignore_index=True)
    data_model.regionprops_df = regionprops_df
    assert isinstance(data_model.regionprops_df, pd.DataFrame)
    assert len(data_model.regionprops_df) == sum(len(df) for df in dfs)
