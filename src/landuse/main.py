"""
Main module for land use modeling analysis.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import json

from .climate_impact import main as climate_impact_analysis
from .logit_estimation import main as logit_estimation
from .marginal_effects import create_marginal_effects_report
from .data_exploration import (
    explore_nri_data,
    analyze_metro_population_changes,
    generate_summary_report
)


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def run_full_analysis(config: Dict[str, Any]):
    """
    Run complete land use analysis pipeline.

    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary with paths and parameters
    """
    print("Starting Land Use Modeling Analysis Pipeline")
    print("=" * 50)

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Data Exploration
    if config.get('run_exploration', True):
        print("\n1. Running Data Exploration...")
        exploration_dir = output_dir / 'exploration'
        exploration_dir.mkdir(exist_ok=True)

        if 'nri_data' in config:
            explore_nri_data(config['nri_data'], str(exploration_dir))

        if 'population_data' in config:
            analyze_metro_population_changes(
                config['population_data'],
                config.get('metro_definition'),
                config.get('years')
            )

    # Step 2: Logit Estimation
    if config.get('run_estimation', True):
        print("\n2. Running Logit Estimation...")
        estimation_dir = output_dir / 'estimation'
        estimation_dir.mkdir(exist_ok=True)

        logit_estimation(
            config['georef_file'],
            config['nr_data_file'],
            config['start_crop_file'],
            config['start_pasture_file'],
            config['start_forest_file'],
            str(estimation_dir)
        )

    # Step 3: Marginal Effects Analysis
    if config.get('run_marginal_effects', True):
        print("\n3. Calculating Marginal Effects...")
        marginal_dir = output_dir / 'marginal_effects'
        marginal_dir.mkdir(exist_ok=True)

        # Load models from estimation
        import pickle
        models_file = output_dir / 'estimation' / 'landuse_models.pkl'
        if models_file.exists():
            with open(models_file, 'rb') as f:
                models_data = pickle.load(f)

            models = {k: v for k, v in models_data.items() if not k.endswith('_data')}
            data = {k: v for k, v in models_data.items() if k.endswith('_data')}

            create_marginal_effects_report(models, data, str(marginal_dir))

    # Step 4: Climate Impact Analysis
    if config.get('run_climate_impact', True):
        print("\n4. Running Climate Impact Analysis...")
        climate_dir = output_dir / 'climate_impact'
        climate_dir.mkdir(exist_ok=True)

        climate_impact_analysis(
            config['model_data_path'],
            config['estimation_data_path'],
            config['cc_impacts_dir'],
            config['shapefile_path'],
            str(climate_dir)
        )

    print("\n" + "=" * 50)
    print(f"Analysis Complete! Results saved to: {output_dir}")


def create_sample_config(output_file: str = "config.json"):
    """Create a sample configuration file."""
    sample_config = {
        "output_dir": "results",
        "run_exploration": True,
        "run_estimation": True,
        "run_marginal_effects": True,
        "run_climate_impact": True,
        "georef_file": "data/forest_georef.csv",
        "nr_data_file": "data/nr_clean_5year_normals.rds",
        "start_crop_file": "data/start_crop.rds",
        "start_pasture_file": "data/start_pasture.rds",
        "start_forest_file": "data/start_forest.rds",
        "nri_data": "data/nri_data.csv",
        "population_data": "data/population.csv",
        "metro_definition": "data/metro_areas.csv",
        "years": [2010, 2011, 2012],
        "model_data_path": "data/crop_forest_levels_2010_2012.rds",
        "estimation_data_path": "data/estimation_data_crop_forest_logit_2010_2012.rds",
        "cc_impacts_dir": "data/cc_impacts",
        "shapefile_path": "shapefiles/conus_county.shp"
    }

    with open(output_file, 'w') as f:
        json.dump(sample_config, f, indent=2)

    print(f"Sample configuration file created: {output_file}")


def main():
    """Main entry point for the land use modeling package."""
    parser = argparse.ArgumentParser(description="Land Use Modeling Analysis")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Full analysis command
    full_parser = subparsers.add_parser('full', help='Run full analysis pipeline')
    full_parser.add_argument('config', help='Path to configuration file')

    # Individual analysis commands
    explore_parser = subparsers.add_parser('explore', help='Run data exploration')
    explore_parser.add_argument('data_file', help='Path to data file')
    explore_parser.add_argument('--output', default='exploration_results',
                               help='Output directory')

    estimate_parser = subparsers.add_parser('estimate', help='Run logit estimation')
    estimate_parser.add_argument('georef', help='Geographic reference file')
    estimate_parser.add_argument('nr_data', help='Net returns data file')
    estimate_parser.add_argument('crop_start', help='Crop start data')
    estimate_parser.add_argument('pasture_start', help='Pasture start data')
    estimate_parser.add_argument('forest_start', help='Forest start data')
    estimate_parser.add_argument('--output', default='estimation_results',
                                help='Output directory')

    marginal_parser = subparsers.add_parser('marginal', help='Calculate marginal effects')
    marginal_parser.add_argument('models_file', help='Path to models pickle file')
    marginal_parser.add_argument('--output', default='marginal_results',
                                help='Output directory')

    climate_parser = subparsers.add_parser('climate', help='Run climate impact analysis')
    climate_parser.add_argument('model_data', help='Model data file')
    climate_parser.add_argument('estimation_data', help='Estimation data file')
    climate_parser.add_argument('cc_impacts_dir', help='Climate impacts directory')
    climate_parser.add_argument('shapefile', help='County shapefile')
    climate_parser.add_argument('--output', default='climate_results',
                               help='Output directory')

    # Create sample config command
    config_parser = subparsers.add_parser('create-config', help='Create sample configuration file')
    config_parser.add_argument('--output', default='config.json',
                              help='Output file name')

    args = parser.parse_args()

    if args.command == 'full':
        config = load_config(args.config)
        run_full_analysis(config)

    elif args.command == 'explore':
        from .data_exploration import explore_nri_data
        explore_nri_data(args.data_file, args.output)

    elif args.command == 'estimate':
        from .logit_estimation import main as estimate_main
        estimate_main(args.georef, args.nr_data, args.crop_start,
                     args.pasture_start, args.forest_start, args.output)

    elif args.command == 'marginal':
        import pickle
        with open(args.models_file, 'rb') as f:
            models_data = pickle.load(f)
        models = {k: v for k, v in models_data.items() if not k.endswith('_data')}
        data = {k: v for k, v in models_data.items() if k.endswith('_data')}
        create_marginal_effects_report(models, data, args.output)

    elif args.command == 'climate':
        from .climate_impact import main as climate_main
        climate_main(args.model_data, args.estimation_data, args.cc_impacts_dir,
                    args.shapefile, args.output)

    elif args.command == 'create-config':
        create_sample_config(args.output)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()