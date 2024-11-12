# src/popupy/cli/main.py
import click
from popupy import __version__
from pathlib import Path

from popupy.utils.config_utils import create_config_template
from ..config.settings import Settings
from ..core.feature_extraction import FeatureExtractor
from ..core.model import Model
from ..core.dasymetric import DasymetricMapper


@click.group(name='popupy')
@click.version_option(version=__version__, prog_name='PopuPy')
@click.pass_context
def cli(ctx):
    """
    PopuPy for geospatial modeling of population distribution.

    A Python toolkit for high-resolution population mapping using machine learning
    and dasymetric techniques.

    For more information, visit: https://popupy.readthedocs.io/
    """
    ctx.ensure_object(dict)


@cli.command()
@click.option('-c', '--config', 'config_file',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration file')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Show detailed information')
@click.option('--no-viz',
              is_flag=True,
              help='Skip visualization')
def run(config_file: str, verbose: bool, no_viz: bool) -> None:
    """Run the complete population modeling workflow."""
    try:
        settings = Settings.from_file(config_file)
        if verbose:
            click.echo(str(settings))

        # Create output directory if it doesn't exist
        output_dir = Path(settings.work_dir) / 'output'
        output_dir.mkdir(exist_ok=True)

        feature_extractor = FeatureExtractor(settings)
        model = Model(settings)
        mapper = DasymetricMapper(settings)

        # Run workflow
        click.echo("Extracting features...")
        features = feature_extractor.extract()

        click.echo("Training model...")
        model.train(features)

        click.echo("Making predictions...")
        predictions = model.predict()

        click.echo("Performing dasymetric mapping...")
        mapper.map(predictions)

        if not no_viz:
            click.echo("Creating visualization...")
            from ..utils.visualization import Visualizer
            visualizer = Visualizer(settings)

            # Validate paths exist
            viz_paths = {
                'mastergrid': settings.mastergrid,
                'prediction': str(output_dir / 'prediction.tif'),
                'normalized_census': str(output_dir / 'normalized_census.tif'),
                'population': str(output_dir / 'dasymetric.tif')
            }

            for name, path in viz_paths.items():
                if not Path(path).exists():
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
            click.echo(f"Visualization saved as '{viz_output}'")

        click.echo("Population modeling completed successfully!")

    except Exception as e:
        click.echo(f"Error during execution: {str(e)}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        raise click.Abort()


@cli.command()
@click.argument('project_dir', type=click.Path())
@click.option('--data-dir', default='data', help='Name of directory containing data files')
@click.option('--prefix', default='test_', help='Prefix for data files')
def init(project_dir: str, data_dir: str, prefix: str):
    """Initialize a new PopuPy project with proper structure."""
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

        click.echo(f"Initialized new PopuPy project in {project_dir}")
        click.echo("\nCreated directory structure:")
        click.echo(f"{project_dir}/")
        click.echo("|-- config.yaml")
        click.echo(f"|-- {data_dir}/")
        click.echo("|   |-- (place your input files here)")
        click.echo("|-- output/")

        click.echo("\nExpected input files:")
        click.echo(f"  {prefix}buildingCount.tif")
        click.echo(f"  {prefix}buildingSurface.tif")
        click.echo(f"  {prefix}buildingVolume.tif")
        click.echo(f"  {prefix}mastergrid.tif")
        click.echo(f"  {prefix}admin3.csv")

    except Exception as e:
        click.echo(f"Error during initialization: {str(e)}", err=True)
        raise click.Abort()



if __name__ == '__main__':
    cli()