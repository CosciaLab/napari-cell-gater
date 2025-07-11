from pathlib import Path

import pandas as pd
import pytest

from cell_gater.utils.csv_df import stack_csv_files, get_gates_from_regionprops_df, get_markers_of_interest


@pytest.fixture
def test_quants_path():
    return Path(__file__).parent / "test_data" / "quants"

@pytest.fixture
def sample_regionprops_df(test_quants_path):
    return stack_csv_files(test_quants_path)

def test_stack_csv_files_success(test_quants_path):
    df = stack_csv_files(test_quants_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "sample_id" in df.columns
    assert set(df["sample_id"].unique()) == {"1", "2", "3", "4", "5"}
    # Check a known value from one of the CSVs to ensure correct stacking
    assert df[df["sample_id"] == "1"].iloc[0]["CellID"] == 5164

def test_stack_csv_files_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty_quants"
    empty_dir.mkdir()
    df = stack_csv_files(empty_dir)
    assert df is None

def test_stack_csv_files_no_csvs(tmp_path):
    no_csv_dir = tmp_path / "no_csv_quants"
    no_csv_dir.mkdir()
    (no_csv_dir / "test.txt").write_text("hello")
    df = stack_csv_files(no_csv_dir)
    assert df is None

def test_get_gates_from_regionprops_df_no_path(sample_regionprops_df):
    markers = ["mean_750_bg", "mean_647_bg"]
    gates = get_gates_from_regionprops_df(None, sample_regionprops_df, markers)
    assert isinstance(gates, pd.DataFrame)
    assert list(gates.index) == markers
    assert set(gates.columns) == set(sample_regionprops_df["sample_id"].unique())

def test_get_gates_from_regionprops_df_with_path(tmp_path, sample_regionprops_df):
    gates_path = tmp_path / "test_gates.csv"
    # Create a dummy gates file with marker as index and sample IDs as columns
    markers = ["mean_750_bg"]
    sample_ids = ["1", "2", "3", "4", "5"]
    dummy_gates = pd.DataFrame(index=markers, columns=sample_ids)
    dummy_gates.loc["mean_750_bg", "1"] = 10.0 # Add some dummy data
    dummy_gates.to_csv(gates_path) # index=True by default

    gates = get_gates_from_regionprops_df(gates_path, sample_regionprops_df, markers)
    assert isinstance(gates, pd.DataFrame)
    assert gates.shape == (1, 5)
    assert gates.index[0] == "mean_750_bg"
    assert list(gates.columns) == sample_ids

def test_get_gates_from_regionprops_df_invalid_path(sample_regionprops_df):
    invalid_path = Path("/non/existent/path/gates.csv")
    markers = ["mean_750_bg"]
    with pytest.raises(AssertionError):
        get_gates_from_regionprops_df(invalid_path, sample_regionprops_df, markers)

def test_get_markers_of_interest_basic(sample_regionprops_df):
    # Assuming 'X_centroid' is a column in your test data
    markers = get_markers_of_interest(sample_regionprops_df, "X_centroid")
    assert "mean_750_bg" in markers
    assert "CellID" not in markers # CellID should be excluded
    assert "X_centroid" not in markers # up_to column should be excluded

def test_get_markers_of_interest_with_subset(sample_regionprops_df):
    markers = get_markers_of_interest(sample_regionprops_df, "X_centroid", subset=(0, 2))
    assert len(markers) == 2 # Should return only 2 markers

def test_get_markers_of_interest_with_nuclear_stains(sample_regionprops_df):
    # Add dummy nuclear stain columns to the dataframe for this test
    df_with_nuclear = sample_regionprops_df.copy()
    df_with_nuclear["DNA_test"] = 1
    df_with_nuclear["DAPI_test"] = 1
    df_with_nuclear["Hoechst_test"] = 1

    markers = get_markers_of_interest(df_with_nuclear, "X_centroid")
    assert "DNA_test" not in markers
    assert "DAPI_test" not in markers
    assert "Hoechst_test" not in markers

