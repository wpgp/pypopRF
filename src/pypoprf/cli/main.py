# src/pypoprf/cli/main.py
import click
import pandas as pd
from pypoprf import __version__
from pathlib import Path

from pypoprf.utils.config_utils import create_config_template
from pypoprf.utils.logger import get_logger
from ..config.settings import Settings
from ..core.feature_extraction import FeatureExtractor
from ..core.model import Model
from ..core.dasymetric import DasymetricMapper
from ..utils.raster import remask_layer

logger = get_logger()

@click.group(name='pypoprf')
@click.version_option(version=__version__, prog_name='pypoprf')
@click.pass_context
def cli(ctx):
    """
    pypopRF for geospatial modeling of population distribution.

    A Python toolkit for high-resolution population mapping using machine learning
    and dasymetric techniques.

    For more information, visit: https://pypoprf.readthedocs.io/
    """
    ctx.ensure_object(dict)

@cli.command()
@click.option('-c', '--config', 'config_file',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration file')
@click.option('-m', '--model', 'model_path',
              type=click.Path(exists=True),
              help='Path to model pickle')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Show detailed information')
@click.option('--no-viz',
              is_flag=True,
              help='Skip visualization')

def run(config_file: str, 
        verbose: bool, 
        no_viz: bool,
        model_path: str) -> None:
    """Run the complete population modeling workflow."""
    logger.info(f"Starting population modeling workflow with config: {config_file}")

    settings = Settings.from_file(config_file)
    logger.debug(f"Settings: {str(settings)}")

    if verbose:
        logger.set_level('DEBUG')

    # Create output directory if it doesn't exist
    output_dir = Path(settings.work_dir) / 'output'
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory created: {output_dir}")

    # Re-mask mastergrid if requested
    if settings.mask:
        logger.info("Remasking mastergrid...")
        outfile = settings.mastergrid.replace('.tif', '_masked.tif')
        remask_layer(settings.mastergrid,
                     settings.mask,
                     1,
                     outfile=outfile,
                     block_size=settings.block_size)
        settings.mastergrid = outfile

    # Constraining mastergrid if requested
    if settings.constrain:
        logger.info("Constraining mastergrid...")
        outfile = settings.mastergrid.replace('.tif', '_constrained.tif')
        remask_layer(settings.mastergrid,
                     settings.constrain,
                     0,
                     outfile=outfile,
                     block_size=settings.block_size)
        settings.constrain = outfile

    feature_extractor = FeatureExtractor(settings)
    # Run workflow
    if model_path:
        logger.info('Loading pre-trained model')
        features = feature_extractor.get_dummy()
        model = Model(settings)
        model.train(features, 
                    model_path=model_path, 
                    scaler_path=model_path.replace('model','scaler'),
                    log_scale=settings.log_scale,
                    save_model=False)
    else:
        logger.info("Starting feature extraction...")
        features = feature_extractor.extract()
        model = Model(settings)
        model.train(features, log_scale=settings.log_scale)

    logger.info("Making predictions...")
    predictions = model.predict(log_scale=settings.log_scale)

    mapper = DasymetricMapper(settings)

    logger.info("Performing dasymetric mapping...")
    mapper.map(predictions)

    if not no_viz:
        logger.info("Creating visualization...")
        from ..utils.visualization import Visualizer
        visualizer = Visualizer(settings)

        viz_paths = {
            'mastergrid': settings.mastergrid,
            'prediction': str(output_dir / 'prediction.tif'),
            'normalized_census': str(output_dir / 'normalized_census.tif'),
            'population': str(output_dir / 'dasymetric.tif')
        }

        for name, path in viz_paths.items():
            if not Path(path).exists():
                error_msg = f"Required file for visualization not found: {name} at {path}"
                logger.error(error_msg)
                raise FileNotFoundError(f"Required file for visualization not found: {name} at {path}")

        # Create visualization
        viz_output = str(output_dir / 'visualization.png')
        visualizer.map_redistribute(
            mastergrid_path=viz_paths['mastergrid'],
            probability_path=viz_paths['prediction'],
            normalize_path=viz_paths['normalized_census'],
            population_path=viz_paths['population'],
            output_path=viz_output,
            vis_params={
                'vmin': [0, 0, 0, 0],
                'vmax': [1300, 250, 1, 250],
                'cmap': 'viridis',
                'titles': ['Zones', 'Probability', 'Normalized Zones', 'Redistributed']
            },
            dpi=300,
            figsize=(15, 5),
            nodata=-99
        )
        logger.info(f"Visualization saved as '{viz_output}'")

    logger.info("Population modeling completed successfully!")

    if verbose:
        import traceback
        click.echo(traceback.format_exc(), err=True)
    raise click.Abort()

@cli.command()
@click.option('-c', '--config', 'config_file',
            type=click.Path(exists=True),
            required=True,
            help='Path to configuration file')
@click.option('-p', '--prediction', 'prediction',
            type=click.Path(exists=True),
            required=True,
            help='Path to prediction layer')
@click.option('-t', '--table', 'table',
            type=click.Path(exists=True),
            required=True,
            help='Path to age-sex structure table')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Show detailed information')

def agesex(config_file: str, 
           prediction: str, 
           table:str, 
           verbose:bool) -> None:
    """Dasymetric redistribution of data with age-sex structure."""

    logger.info(f"Starting age-sex redistribution with config: {config_file}")

    settings = Settings.from_file(config_file)
    logger.debug(f"Settings: {str(settings)}")

    if verbose:
        logger.set_level('DEBUG')

    # Create output directory if it doesn't exist
    output_dir = Path(settings.work_dir) / 'output'
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory created: {output_dir}")

    # Re-mask mastergrid if requested
    if settings.mask:
        logger.info("Remasking mastergrid...")
        outfile = settings.mastergrid.replace('.tif', '_masked.tif')
        settings.mastergrid = outfile
        if not Path(outfile).is_file():
            remask_layer(settings.mastergrid,
                         settings.mask,
                         1,
                         outfile=outfile,
                         block_size=settings.block_size)                

    # Constraining mastergrid if requested
    if settings.constrain:
        logger.info("Constraining mastergrid...")
        outfile = settings.mastergrid.replace('.tif', '_constrained.tif')
        settings.constrain = outfile
        if not Path(outfile).is_file():
            remask_layer(settings.mastergrid,
                        settings.constrain,
                        0,
                        outfile=outfile,
                        block_size=settings.block_size)
            
    mapper = DasymetricMapper(settings)
    mapper.map_agesex(prediction, table)

    logger.info("Age-sex redistribution completed successfully!")

    if verbose:
        import traceback
        click.echo(traceback.format_exc(), err=True)
    raise click.Abort()

@cli.command()
@click.argument('project_dir', type=click.Path())
@click.option('--data-dir', default='data', help='Name of directory containing data files')
@click.option('--prefix', default='test_', help='Prefix for data files')
def init(project_dir: str, data_dir: str, prefix: str):
    """Initialize a new pypopRF project with proper structure."""
    try:
        # Create project directory
        project_path = Path(project_dir).resolve()
        project_path.mkdir(parents=True, exist_ok=True)

        # Create directories
        data_path = project_path / data_dir
        data_path.mkdir(exist_ok=True)

        output_path = project_path / 'output'
        output_path.mkdir(exist_ok=True)

        # Create config
        config_path = project_path / "config.yaml"
        create_config_template(
            output_path=config_path,
            data_dir=data_dir,
            prefix=prefix
        )

        logger.info(f"Initialized new pypopRF project in {project_dir}")
        logger.info("\nCreated directory structure:")
        logger.info(f"{project_dir}/")
        logger.info("|-- config.yaml")
        logger.info(f"|-- {data_dir}/")
        logger.info("|   |-- (place your input files here)")
        logger.info("|-- output/")

        logger.info("\nExpected input files:")
        logger.info(f"  {prefix}buildingCount.tif")
        logger.info(f"  {prefix}buildingSurface.tif")
        logger.info(f"  {prefix}buildingVolume.tif")
        logger.info(f"  {prefix}mastergrid.tif")
        logger.info(f"  {prefix}admin3.csv")

    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise click.Abort()


if __name__ == '__main__':
    cli()
