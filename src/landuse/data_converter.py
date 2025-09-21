"""
Data conversion utilities for preparing data for land use modeling.
Helps convert various data formats to the expected structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings


class DataConverter:
    """Convert and validate data for land use modeling pipeline."""

    # Land use code mappings
    LANDUSE_CODES = {
        'crop': 1, 'cropland': 1, 'cultivated': 1, 'agriculture': 1,
        'pasture': 2, 'grassland': 2, 'rangeland': 2, 'grazing': 2,
        'forest': 3, 'woodland': 3, 'trees': 3, 'timber': 3,
        'urban': 4, 'developed': 4, 'built': 4, 'residential': 4,
        'commercial': 4, 'industrial': 4
    }

    # Region mappings
    REGION_CODES = {
        'north': 'NO', 'northeast': 'NO', 'midwest': 'MW',
        'south': 'SO', 'southeast': 'SO', 'southwest': 'SO',
        'west': 'WE', 'northwest': 'WE', 'pacific': 'WE'
    }

    @staticmethod
    def convert_fips(df: pd.DataFrame, fips_column: str = 'fips') -> pd.DataFrame:
        """
        Ensure FIPS codes are properly formatted as integers.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fips_column : str
            Name of FIPS column

        Returns:
        --------
        pd.DataFrame
            DataFrame with corrected FIPS codes
        """
        df = df.copy()

        if fips_column not in df.columns:
            raise ValueError(f"Column '{fips_column}' not found in dataframe")

        # Handle string FIPS codes
        if df[fips_column].dtype == 'object':
            # Remove any leading zeros and convert to int
            df[fips_column] = df[fips_column].str.strip().str.lstrip('0')
            df[fips_column] = pd.to_numeric(df[fips_column], errors='coerce')

        # Ensure integer type
        df[fips_column] = df[fips_column].astype('Int64')  # Nullable integer

        # Validate FIPS codes (should be 1-5 digits)
        invalid_fips = df[(df[fips_column] < 1) | (df[fips_column] > 99999)][fips_column]
        if not invalid_fips.empty:
            warnings.warn(f"Found {len(invalid_fips)} invalid FIPS codes")

        return df

    @staticmethod
    def convert_landuse_codes(df: pd.DataFrame,
                             landuse_column: str,
                             custom_mapping: Optional[Dict] = None) -> pd.DataFrame:
        """
        Convert land use text labels to numeric codes.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        landuse_column : str
            Column with land use labels
        custom_mapping : Dict, optional
            Custom mapping of labels to codes

        Returns:
        --------
        pd.DataFrame
            DataFrame with numeric land use codes
        """
        df = df.copy()
        mapping = custom_mapping or DataConverter.LANDUSE_CODES

        if df[landuse_column].dtype in ['int64', 'float64']:
            # Already numeric
            return df

        # Convert text to lowercase for matching
        df[f'{landuse_column}_original'] = df[landuse_column]
        df[landuse_column] = df[landuse_column].str.lower().str.strip()

        # Map to numeric codes
        df[f'{landuse_column}_code'] = df[landuse_column].map(mapping)

        # Check for unmapped values
        unmapped = df[df[f'{landuse_column}_code'].isna()][landuse_column].unique()
        if len(unmapped) > 0:
            warnings.warn(f"Unmapped land use values: {unmapped}")

        return df

    @staticmethod
    def prepare_georef(df: pd.DataFrame,
                      fips_col: str = 'fips',
                      state_col: Optional[str] = None,
                      county_col: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare geographic reference data.

        Parameters:
        -----------
        df : pd.DataFrame
            Input geographic data
        fips_col : str
            FIPS column name
        state_col : str, optional
            State column (to create FIPS if missing)
        county_col : str, optional
            County column (to create FIPS if missing)

        Returns:
        --------
        pd.DataFrame
            Formatted geographic reference
        """
        df = df.copy()

        # Create FIPS from state and county if needed
        if fips_col not in df.columns and state_col and county_col:
            df[fips_col] = df[state_col].astype(str).str.zfill(2) + \
                          df[county_col].astype(str).str.zfill(3)
            df[fips_col] = df[fips_col].astype(int)

        # Convert FIPS to proper format
        df = DataConverter.convert_fips(df, fips_col)

        # Ensure required columns
        if 'county_fips' not in df.columns:
            df['county_fips'] = df[fips_col]

        # Add region if missing
        if 'region' not in df.columns:
            # Simple assignment based on state code
            df['state_code'] = df[fips_col] // 1000
            df['region'] = df['state_code'].apply(lambda x: DataConverter._assign_region(x))

        if 'subregion' not in df.columns:
            df['subregion'] = df['region'].apply(lambda x: x + 'E' if x else None)

        return df[[fips_col, 'county_fips', 'region', 'subregion']]

    @staticmethod
    def _assign_region(state_code: int) -> str:
        """Assign region based on state FIPS code."""
        # Simplified regional assignment
        if state_code in range(1, 25):  # Northeast/North
            return 'NO'
        elif state_code in range(25, 40):  # South
            return 'SO'
        elif state_code in range(40, 50):  # Midwest
            return 'MW'
        else:  # West
            return 'WE'

    @staticmethod
    def prepare_net_returns(df: pd.DataFrame,
                           value_cols: Dict[str, str],
                           fips_col: str = 'fips',
                           year_col: str = 'year') -> pd.DataFrame:
        """
        Prepare net returns data with proper column names.

        Parameters:
        -----------
        df : pd.DataFrame
            Input net returns data
        value_cols : Dict[str, str]
            Mapping of {target_name: source_column}
            e.g., {'nr_cr': 'crop_returns', 'nr_ps': 'pasture_returns'}
        fips_col : str
            FIPS column name
        year_col : str
            Year column name

        Returns:
        --------
        pd.DataFrame
            Formatted net returns data
        """
        df = df.copy()

        # Convert FIPS
        df = DataConverter.convert_fips(df, fips_col)

        # Rename columns
        rename_dict = {v: k for k, v in value_cols.items()}
        df = df.rename(columns=rename_dict)

        # Ensure all required columns exist
        required = ['nr_cr', 'nr_ps', 'nr_fr', 'nr_ur']
        for col in required:
            if col not in df.columns:
                warnings.warn(f"Missing {col}, filling with zeros")
                df[col] = 0

        # Select and order columns
        cols = [fips_col, year_col] + required
        optional = ['nrmean_cr', 'nrmean_ps', 'nrmean_fr', 'nrmean_ur',
                   'nrchange_cr', 'nrchange_ps', 'nrchange_fr', 'nrchange_ur']

        for col in optional:
            if col in df.columns:
                cols.append(col)

        return df[cols]

    @staticmethod
    def prepare_transitions(df: pd.DataFrame,
                          start_col: str = 'start_landuse',
                          end_col: str = 'end_landuse',
                          fips_col: str = 'fips',
                          year_col: str = 'year',
                          lcc_col: Optional[str] = None,
                          weight_col: Optional[str] = None,
                          id_col: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare land use transition data.

        Parameters:
        -----------
        df : pd.DataFrame
            Input transition data
        start_col : str
            Starting land use column
        end_col : str
            Ending land use column
        fips_col : str
            FIPS column
        year_col : str
            Year column
        lcc_col : str, optional
            Land capability class column
        weight_col : str, optional
            Weight/expansion factor column
        id_col : str, optional
            Plot/observation ID column

        Returns:
        --------
        pd.DataFrame
            Formatted transition data
        """
        df = df.copy()

        # Convert FIPS
        df = DataConverter.convert_fips(df, fips_col)

        # Convert land use codes
        df = DataConverter.convert_landuse_codes(df, start_col)
        df = DataConverter.convert_landuse_codes(df, end_col)

        # Rename to expected names
        df['startuse'] = df[f'{start_col}_code']
        df['enduse'] = df[f'{end_col}_code']

        # Handle LCC
        if lcc_col and lcc_col in df.columns:
            df['lcc'] = df[lcc_col]
        else:
            # Assign random LCC if missing
            warnings.warn("No LCC column found, assigning random values 1-4")
            df['lcc'] = np.random.randint(1, 5, size=len(df))

        # Handle weights
        if weight_col and weight_col in df.columns:
            df['xfact'] = df[weight_col]
        else:
            warnings.warn("No weight column found, using uniform weights")
            df['xfact'] = 1.0

        # Handle ID
        if id_col and id_col in df.columns:
            df['riad_id'] = df[id_col]
        else:
            df['riad_id'] = df.index.astype(str)

        # Select final columns
        cols = [fips_col, year_col, 'riad_id', 'startuse', 'enduse', 'lcc', 'xfact']
        return df[cols]

    @staticmethod
    def split_by_start_use(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split transition data by starting land use.

        Parameters:
        -----------
        df : pd.DataFrame
            Combined transition data with 'startuse' column

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary with keys 'crop', 'pasture', 'forest', 'urban'
        """
        landuse_names = {1: 'crop', 2: 'pasture', 3: 'forest', 4: 'urban'}
        result = {}

        for code, name in landuse_names.items():
            subset = df[df['startuse'] == code].copy()
            if not subset.empty:
                result[name] = subset
            else:
                warnings.warn(f"No transitions found starting from {name}")

        return result


def convert_existing_data(input_dir: str,
                         output_dir: str,
                         config: Optional[Dict] = None) -> None:
    """
    Convert existing data files to the expected format.

    Parameters:
    -----------
    input_dir : str
        Directory with input data files
    output_dir : str
        Directory for output files
    config : Dict, optional
        Configuration for column mappings

    Example config:
    {
        "georef": {
            "file": "counties.csv",
            "fips_col": "FIPS",
            "state_col": "STATE",
            "county_col": "COUNTY"
        },
        "net_returns": {
            "file": "returns.csv",
            "columns": {
                "nr_cr": "crop_revenue",
                "nr_ps": "pasture_revenue",
                "nr_fr": "forest_revenue",
                "nr_ur": "urban_value"
            }
        },
        "transitions": {
            "file": "landuse_changes.csv",
            "start_col": "from_use",
            "end_col": "to_use",
            "lcc_col": "land_quality",
            "weight_col": "sample_weight"
        }
    }
    """
    converter = DataConverter()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = config or {}

    # Convert geographic reference
    if 'georef' in config:
        georef_config = config['georef']
        georef_file = input_path / georef_config['file']
        if georef_file.exists():
            print(f"Converting geographic reference: {georef_file}")
            georef = pd.read_csv(georef_file)
            georef = converter.prepare_georef(
                georef,
                fips_col=georef_config.get('fips_col', 'fips'),
                state_col=georef_config.get('state_col'),
                county_col=georef_config.get('county_col')
            )
            georef.to_csv(output_path / "forest_georef.csv", index=False)
            print(f"  ✓ Saved forest_georef.csv ({len(georef)} counties)")

    # Convert net returns
    if 'net_returns' in config:
        nr_config = config['net_returns']
        nr_file = input_path / nr_config['file']
        if nr_file.exists():
            print(f"Converting net returns: {nr_file}")
            nr_data = pd.read_csv(nr_file)
            nr_data = converter.prepare_net_returns(
                nr_data,
                value_cols=nr_config['columns'],
                fips_col=nr_config.get('fips_col', 'fips'),
                year_col=nr_config.get('year_col', 'year')
            )
            nr_data.to_csv(output_path / "nr_clean_5year_normals.csv", index=False)
            print(f"  ✓ Saved nr_clean_5year_normals.csv ({len(nr_data)} observations)")

    # Convert transitions
    if 'transitions' in config:
        trans_config = config['transitions']
        trans_file = input_path / trans_config['file']
        if trans_file.exists():
            print(f"Converting transitions: {trans_file}")
            trans_data = pd.read_csv(trans_file)
            trans_data = converter.prepare_transitions(
                trans_data,
                start_col=trans_config.get('start_col', 'start_landuse'),
                end_col=trans_config.get('end_col', 'end_landuse'),
                fips_col=trans_config.get('fips_col', 'fips'),
                year_col=trans_config.get('year_col', 'year'),
                lcc_col=trans_config.get('lcc_col'),
                weight_col=trans_config.get('weight_col'),
                id_col=trans_config.get('id_col')
            )

            # Split by starting use
            splits = converter.split_by_start_use(trans_data)
            for name, data in splits.items():
                data.to_csv(output_path / f"start_{name}.csv", index=False)
                print(f"  ✓ Saved start_{name}.csv ({len(data)} transitions)")

    print(f"\nConversion complete! Files saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Data Converter Example")
    print("-" * 40)

    # Create sample data to demonstrate conversion
    sample_georef = pd.DataFrame({
        'FIPS': ['01001', '01003', '06001'],
        'STATE': [1, 1, 6],
        'COUNTY': [1, 3, 1],
        'NAME': ['County A', 'County B', 'County C']
    })

    sample_returns = pd.DataFrame({
        'fips': [1001, 1003, 6001],
        'year': [2010, 2010, 2010],
        'crop_revenue': [250, 280, 300],
        'pasture_revenue': [80, 90, 85],
        'forest_revenue': [40, 45, 50],
        'urban_value': [5000, 5500, 6000]
    })

    sample_transitions = pd.DataFrame({
        'fips': [1001, 1001, 1003, 6001],
        'year': [2010, 2010, 2010, 2010],
        'from_use': ['cropland', 'forest', 'pasture', 'cropland'],
        'to_use': ['cropland', 'cropland', 'urban', 'pasture'],
        'land_quality': [2, 3, 4, 2],
        'sample_weight': [100, 150, 120, 110]
    })

    # Convert samples
    converter = DataConverter()

    print("\n1. Converting FIPS codes:")
    georef = converter.prepare_georef(sample_georef, fips_col='FIPS',
                                     state_col='STATE', county_col='COUNTY')
    print(georef)

    print("\n2. Converting net returns:")
    nr = converter.prepare_net_returns(
        sample_returns,
        value_cols={
            'nr_cr': 'crop_revenue',
            'nr_ps': 'pasture_revenue',
            'nr_fr': 'forest_revenue',
            'nr_ur': 'urban_value'
        }
    )
    print(nr)

    print("\n3. Converting transitions:")
    trans = converter.prepare_transitions(
        sample_transitions,
        start_col='from_use',
        end_col='to_use',
        lcc_col='land_quality',
        weight_col='sample_weight'
    )
    print(trans)

    print("\n4. Splitting by start use:")
    splits = converter.split_by_start_use(trans)
    for name, data in splits.items():
        print(f"\n{name} transitions:")
        print(data)