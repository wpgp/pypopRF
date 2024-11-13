import matplotlib.pyplot as plt
import rasterio
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple

from ..config.settings import Settings


class Visualizer:
    """Handle visualization of population modeling results."""

    def __init__(self, settings: Settings):
        """
        Initialize visualizer.

        Args:
            settings: pypopRF settings instance containing configuration
        """
        self.settings = settings
        self.output_dir = Path(settings.work_dir) / 'output'

    def map_redistribute(
            self,
            mastergrid_path: str,
            probability_path: str,
            normalize_path: str,
            population_path: str,
            output_path: Optional[str] = None,
            vis_params: Optional[Dict] = None,
            dpi: int = 300,
            figsize: Tuple[int, int] = (15, 5),
            nodata: float = -99
    ) -> None:
        """
        Create visualization of population redistribution process.

        Args:
            mastergrid_path: Path to mastergrid raster file
            probability_path: Path to probability surface raster
            normalize_path: Path to normalized census raster
            population_path: Path to final population raster
            output_path: Path to save the visualization (optional)
            vis_params: Dictionary of visualization parameters with format:
                {
                    'vmin': [min1, min2, min3, min4],  # Min values for each panel
                    'vmax': [max1, max2, max3, max4],  # Max values for each panel
                    'cmap': str or [str, str, str, str],  # Colormap(s) to use
                    'titles': [str, str, str, str]  # Panel titles
                }
            dpi: Resolution of output image
            figsize: Size of the figure in inches (width, height)
            nodata: Value to be masked in the visualization
        """
        # Set default visualization parameters
        if vis_params is None:
            vis_params = {
                'vmin': [0, 0, 0, 0],
                'vmax': [1300, 250, 1, 250],
                'cmap': 'viridis',
                'titles': ['Zones', 'Probability', 'Normalized Zones', 'Redistributed']
            }

        # Load and mask rasters
        with rasterio.open(mastergrid_path) as src:
            mastergrid_data = src.read(1)
            mastergrid_data = np.ma.masked_where(mastergrid_data == nodata, mastergrid_data)
            transform = src.transform
            bounds = src.bounds

        with rasterio.open(probability_path) as src:
            prob_data = src.read(1)
            prob_data = np.ma.masked_where(prob_data == nodata, prob_data)

        with rasterio.open(normalize_path) as src:
            norm_data = src.read(1)
            norm_data = np.ma.masked_where(norm_data == nodata, norm_data)

        with rasterio.open(population_path) as src:
            pop_data = src.read(1)
            pop_data = np.ma.masked_where(pop_data == nodata, pop_data)

        # Create figure and subplots
        fig, ax = plt.subplots(1, 4, figsize=figsize, dpi=dpi)

        # Set consistent extent for all plots
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        # Create list of data and corresponding axes for plotting
        data_list = [mastergrid_data, prob_data, norm_data, pop_data]
        images = []

        # Plot each dataset
        for i, data in enumerate(data_list):
            images.append(ax[i].imshow(
                data,
                cmap=vis_params['cmap'],
                vmin=vis_params['vmin'][i],
                vmax=vis_params['vmax'][i],
                extent=extent
            ))

            # Set panel styling
            ax[i].set_title(vis_params['titles'][i])
            ax[i].set_axis_off()
            ax[i].set_aspect('equal')

        # Adjust layout for better appearance
        plt.tight_layout()

        # Handle output
        if output_path:
            # Ensure proper file extension
            if not output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                output_path = output_path + '.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
            plt.close()
        else:
            plt.show()
