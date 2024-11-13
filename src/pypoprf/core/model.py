# src/pypoprf/core/model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import rasterio
import threading
import concurrent.futures
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance

from ..config.settings import Settings
from ..utils.joblib_manager import joblib_resources
from ..utils.matplotlib_utils import with_non_interactive_matplotlib
from ..utils.raster import progress_bar


class Model:
    """
    Population prediction model handler.

    This class manages the training, feature selection, and prediction processes
    for population modeling using Random Forest regression.

    Attributes:
        settings (Settings): Configuration settings for the model
        model (RandomForestRegressor): Trained Random Forest model
        scaler (RobustScaler): Fitted feature scaler
        feature_names (np.ndarray): Names of selected features
        target_mean (float): Mean of target variable for normalization
        output_dir (Path): Directory for saving outputs
    """

    def __init__(self, settings: Settings):
        """
        Initialize model handler.

        Args:
            settings: pypopRF settings instance
        """
        self.settings = settings
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.target_mean = None

        # Create output directory
        self.output_dir = Path(settings.work_dir) / 'output'
        self.output_dir.mkdir(exist_ok=True)

    def train(self,
              data: pd.DataFrame,
              model_path: Optional[str] = None,
              scaler_path: Optional[str] = None,
              save_model: bool = True) -> None:
        """
        Train Random Forest model for population prediction.

        Args:
            data: DataFrame containing features and target variables
                 Must include 'id', 'pop', 'dens' columns
            model_path: Optional path to load pretrained model
            scaler_path: Optional path to load fitted scaler
            save_model: Whether to save model after training

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If model loading fails
        """

        data = data.dropna()
        drop_cols = np.intersect1d(data.columns.values, ['id', 'pop', 'dens'])
        X = data.drop(columns=drop_cols).copy()
        y = data['dens'].values
        self.target_mean = y.mean()
        self.feature_names = X.columns.values

        if scaler_path is None:
            self.scaler = RobustScaler()
            self.scaler.fit(X)
        else:
            with joblib_resources():
                self.scaler = joblib.load(scaler_path)

        if model_path is None:
            X_scaled = self.scaler.transform(X)
            self.model = RandomForestRegressor(n_estimators=500)

            with joblib_resources():
                importances, selected = self._select_features(X_scaled, y)

            X = X[selected]
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            print(f'Training RF model with {self.model.n_estimators} trees')
            self.model.fit(X_scaled, y)
        else:
            with joblib_resources():
                self.model = joblib.load(model_path)
        with joblib_resources():
            self._calculate_cv_scores(X_scaled, y)

        if save_model:
            with joblib_resources():
                self._save_model()

    def _select_features(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         limit: float = 0.05,
                         plot: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Select features based on importance using permutation importance.
        """
        print('Selecting predictive features')

        names = self.feature_names
        ymean = self.target_mean

        # Fit model
        model = self.model.fit(X, y)

        result = permutation_importance(
            model, X, y,
            n_repeats=10,
            n_jobs=2,
            scoring='neg_root_mean_squared_error'
        )

        sorted_idx = result.importances_mean.argsort()

        importances = pd.DataFrame(
            result.importances[sorted_idx].T / ymean,
            columns=names,
        )

        selected = importances.columns.values[importances.mean(axis=0) > limit]

        # Create visualization if requested
        if plot:
            self._plot_feature_importance(importances, limit)

        print(f'Selected {len(selected)} features out of {len(names)} features')
        print(selected.tolist())

        return importances, selected

    @with_non_interactive_matplotlib
    def _plot_feature_importance(self,
                                 importance_df: pd.DataFrame,
                                 limit: float) -> None:
        """
        Create box plot visualization of feature importances.

        Args:
            importance_df: DataFrame with feature importances
            limit: Threshold line to display
        """

        sy = importance_df.shape[1] * 0.25 + 0.5
        fig, ax = plt.subplots(1, 1, figsize=(4, sy), dpi=90)

        importance_df.plot.box(
            vert=False,
            whis=5,
            ax=ax,
            color='k',
            sym='.k'
        )

        ax.axvline(x=limit, color='k', linestyle='--', lw=0.5)
        ax.set_xlabel('Decrease in nRMSE')

        plt.tight_layout()
        save_path = Path(self.settings.work_dir) / 'output' / 'feature_selection.png'
        plt.savefig(save_path)
        plt.close()

    def _calculate_cv_scores(self,
                             X_scaled: np.ndarray,
                             y: np.ndarray,
                             cv: int = 10) -> None:
        """Calculate and print cross-validation scores."""

        print('\nComputing CV scores...')

        scoring = {'r2': (100, 'R2'),
                   'neg_root_mean_squared_error': (-1, 'RMSE'),
                   'neg_mean_absolute_error': (-1, 'MAE')
                   }

        scores = cross_validate(
            self.model, X_scaled, y,
            cv=cv,
            scoring=list(scoring.keys()),
            return_train_score=True
        )

        for k in ['neg_root_mean_squared_error', 'neg_mean_absolute_error']:
            scoring['n' + k] = (-100, 'n' + scoring[k][1])
            scores['test_n' + k] = scores['test_' + k] / self.target_mean
            scores['train_n' + k] = scores['train_' + k] / self.target_mean

        print('\nTraining scores:')
        for k in scoring:
            a = scoring[k][0] * scores[f'train_{k}']
            print(f'{scoring[k][1]:6s}: {a.mean():.2f}')

        print('\nTesting scores:')
        for k in scoring:
            a = scoring[k][0] * scores[f'test_{k}']
            print(f'{scoring[k][1]:6s}: {a.mean():.2f}')

    @with_non_interactive_matplotlib
    def predict(self) -> str:
        """
        Generate predictions using trained model.

        Returns:
            str: Path to output prediction raster file

        Raises:
            RuntimeError: If model is not trained
            FileNotFoundError: If input rasters are missing
        """

        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call train() first.")

        with joblib_resources():
            src = {}
            for k in self.settings.covariate:
                src[k] = rasterio.open(self.settings.covariate[k], 'r')

            try:
                # Open mastergrid
                mst = rasterio.open(self.settings.mastergrid, 'r')

                # Get profile from mastergrid
                profile = mst.profile.copy()
                profile.update({
                    'dtype': 'float32',
                    'blockxsize': self.settings.blocksize[0],
                    'blockysize': self.settings.blocksize[1],
                })

                # Setup locks
                reading_lock = threading.Lock()
                writing_lock = threading.Lock()

                names = self.feature_names
                outfile = Path(self.settings.output_dir) / 'prediction.tif'

                with rasterio.open(outfile, 'w', **profile) as dst:
                    def process(window):
                        df = pd.DataFrame()
                        with reading_lock:
                            for s in src:
                                arr = src[s].read(window=window)[0, :, :]
                                df[s + '_avg'] = arr.flatten()

                        df = df[names]

                        # Make predictions
                        sx = self.scaler.transform(df)
                        yp = self.model.predict(sx)
                        res = yp.reshape(arr.shape)

                        with writing_lock:
                            dst.write(res, window=window, indexes=1)

                    if self.settings.by_block:
                        windows = [window for ij, window in dst.block_windows()]
                        with concurrent.futures.ThreadPoolExecutor(
                                max_workers=self.settings.max_workers
                        ) as executor:
                            list(progress_bar(
                                executor.map(process, windows),
                                self.settings.show_progress,
                                len(windows),
                                desc="Prediction"
                            ))

            finally:
                mst.close()
                for s in src:
                    src[s].close()

        return str(outfile)

    def _save_model(self) -> None:
        """Save model and scaler to disk."""
        model_path = self.output_dir / 'model.pkl.gz'
        scaler_path = self.output_dir / 'scaler.pkl.gz'

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self,
                   model_path: str,
                   scaler_path: str) -> None:
        """
        Load saved model and scaler.

        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = self.scaler.get_feature_names_out()
