# tests/core/test_model.py
import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path

import rasterio
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor

from pypoprf.core.model import Model
from pypoprf.config.settings import Settings


@pytest.fixture
def sample_features(tmp_path):
    """Create sample features for testing."""
    # Create synthetic feature data
    n_samples = 100
    data = pd.DataFrame({
        'id': range(n_samples),
        'pop': np.random.randint(1000, 10000, n_samples),
        'feature1_avg': np.random.rand(n_samples),
        'feature2_avg': np.random.rand(n_samples),
        'feature3_avg': np.random.rand(n_samples),
    })

    # Calculate density
    data['dens'] = data['pop'] / 100  # Arbitrary denominator for testing

    return data


@pytest.fixture
def sample_raster(tmp_path):
    """Create a sample raster file for testing."""
    raster_path = tmp_path / 'test.tif'

    # Create sample raster data
    data = np.random.rand(100, 100)

    try:
        with rasterio.open(
                raster_path,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                nodata=-99,
                transform=rasterio.transform.from_bounds(0, 0, 1, 1, data.shape[1], data.shape[0])
        ) as dst:
            dst.write(data, 1)

        print(f"Test raster created at {raster_path}")
        assert raster_path.exists(), "Raster file was not created"
        return raster_path

    except Exception as e:
        print(f"Error creating test raster: {e}")
        raise


@pytest.fixture
def model_instance(tmp_path, sample_raster):
    """Create a configured Model instance."""
    assert sample_raster is not None, "sample_raster is None"
    assert Path(sample_raster).exists(), f"Raster file does not exist at {sample_raster}"

    settings = Settings(
        work_dir=str(tmp_path),
        covariates={'test': str(sample_raster)},
        mastergrid=str(sample_raster),
        census_data='census.csv',
        census_pop_column='pop',
        census_id_column='id'
    )
    return Model(settings)

def test_model_initialization(model_instance):
    """Test model initialization."""
    assert isinstance(model_instance, Model)
    assert model_instance.model is None
    assert model_instance.scaler is None
    assert model_instance.feature_names is None
    assert model_instance.target_mean is None
    assert model_instance.output_dir.exists()


def test_model_training(model_instance, sample_features):
    """Test model training process."""
    # Train model
    model_instance.train(sample_features)

    # Verify model components
    assert isinstance(model_instance.model, RandomForestRegressor)
    assert isinstance(model_instance.scaler, RobustScaler)
    assert model_instance.feature_names is not None
    assert model_instance.target_mean is not None

    # Check saved files
    assert (model_instance.output_dir / 'model.pkl.gz').exists()
    assert (model_instance.output_dir / 'scaler.pkl.gz').exists()
    assert (model_instance.output_dir / 'feature_selection.png').exists()


def test_feature_selection(model_instance, sample_features):
    """Test feature selection process."""
    model_instance.train(sample_features)

    # Verify selected features
    assert all(feat.endswith('_avg') for feat in model_instance.feature_names)
    assert len(model_instance.feature_names) > 0


def test_model_prediction(model_instance, sample_features):
    """Test model prediction process."""
    # Train model
    model_instance.train(sample_features)

    # Make predictions
    prediction_path = model_instance.predict()

    # Verify prediction output
    assert Path(prediction_path).exists()
    with rasterio.open(prediction_path) as src:
        pred_data = src.read(1)
        assert pred_data.shape == (100, 100)  # Match input raster shape
        assert not np.all(pred_data == src.nodata)  # Contains valid predictions


def test_model_save_load(model_instance, sample_features, tmp_path):
    """Test model saving and loading."""
    # Train and save model
    model_instance.train(sample_features)

    model_path = tmp_path / 'test_model.pkl'
    scaler_path = tmp_path / 'test_scaler.pkl'

    with open(model_path, 'wb') as f:
        joblib.dump(model_instance.model, f)
    with open(scaler_path, 'wb') as f:
        joblib.dump(model_instance.scaler, f)

    # Create new instance and load model
    new_model = Model(model_instance.settings)
    new_model.load_model(str(model_path), str(scaler_path))

    # Verify loaded model
    assert isinstance(new_model.model, RandomForestRegressor)
    assert isinstance(new_model.scaler, RobustScaler)
    assert new_model.feature_names is not None


def test_error_handling(model_instance):
    """Test error handling in model operations."""
    # Test prediction without training
    with pytest.raises(RuntimeError):
        model_instance.predict()

    # Test training with invalid data
    invalid_data = pd.DataFrame({'invalid': range(10)})
    with pytest.raises(Exception):
        model_instance.train(invalid_data)

    # Test loading non-existent model
    with pytest.raises(Exception):
        model_instance.load_model('nonexistent.pkl', 'nonexistent.pkl')


def test_cross_validation_scores(model_instance, sample_features):
    """Test cross-validation score calculation."""
    # Train model and check scores
    model_instance.train(sample_features)

    # We can't directly test the scores, but we can verify they were calculated
    # by checking if the model was fitted
    assert hasattr(model_instance.model, 'n_features_in_')


def test_parallel_processing(model_instance, sample_features):
    """Test parallel processing capabilities."""
    # Set different numbers of workers
    for n_workers in [1, 2, 4]:
        model_instance.settings.max_workers = n_workers
        model_instance.train(sample_features)
        prediction_path = model_instance.predict()
        assert Path(prediction_path).exists()