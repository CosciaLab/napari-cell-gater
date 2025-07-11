
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

from cell_gater.model.data_model import DataModel
from cell_gater.utils.csv_df import create_dataframe_from_csv_files


def test_data_model_population():
    # Create a DataModel instance
    model = DataModel()

    # Define the paths to the test data
    quantification_dir = Path("test_data/quants")
    
    # Create the dataframe
    df = create_dataframe_from_csv_files(quantification_dir)

    # Set the regionprops_df attribute
    model.regionprops_df = df

    # Check that the samples are populated correctly
    assert model.samples == ["1", "2", "3", "4", "5"]

    # Check that the regionprops_df is populated correctly
    assert isinstance(model.regionprops_df, pd.DataFrame)
    assert not model.regionprops_df.empty

def test_data_model_events():
    # Create a DataModel instance
    model = DataModel()

    # Create a mock listener
    listener = Mock()

    # Connect the listener to the regionprops_df event
    model.events.regionprops_df.connect(listener)

    # Set the regionprops_df attribute
    model.regionprops_df = pd.DataFrame()

    # Check that the listener was called
    listener.assert_called_once()
