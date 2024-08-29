#!/bin/bash

# Directory containing the ENVI binary raster files
input_dir="/Users/mihiarc/Work/geodata-econ/rpa-landuse-proj/data/spatial-proj"

# Directory where the converted GeoTIFF files will be saved
output_dir="/Users/mihiarc/Work/geodata-econ/rpa-landuse-proj/data/spatial-proj-geotiff"

# Check if the output directory exists, if not, create it
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Loop through each .hdr file in the input directory
for hdr_file in "$input_dir"/*.hdr; do
  # Extract the base name of the file (without extension)
  base_name=$(basename "$hdr_file" .hdr)
  
  # Define the corresponding data file path
  data_file="$input_dir/$base_name"
  
  # Check if the data file exists
  if [ ! -f "$data_file" ]; then
    echo "Data file $data_file does not exist. Skipping..."
    continue
  fi

  # Define the output file path
  output_file="$output_dir/${base_name}.tif"
  
  # Convert the ENVI binary raster to GeoTIFF using gdal_translate
  gdal_translate -of GTiff "$data_file" "$output_file"
  
  # Check if the conversion was successful
  if [ $? -eq 0 ]; then
    echo "Successfully converted $data_file to $output_file"
    # Delete the ENVI files
    echo "Deleting $data_file and $hdr_file"
    rm "$data_file" "$hdr_file"
  else
    echo "Failed to convert $data_file"
  fi
done