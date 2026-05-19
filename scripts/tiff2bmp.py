import cv2
import click
import os
from pathlib import Path
import numpy as np


@click.command()
@click.argument(
    "input_folder", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument("min_value", type=int)
@click.argument("max_value", type=int)
def convert_tiff_to_bmp(input_folder, min_value, max_value):
    """
    Convert 16-bit TIFF images to 8-bit BMP images.

    INPUT_FOLDER: Path to folder containing TIFF images
    MIN_VALUE: Minimum value for contrast stretching
    MAX_VALUE: Maximum value for contrast stretching
    """
    input_path = Path(input_folder)
    output_path = input_path / "BMP"
    output_path.mkdir(exist_ok=True)

    # Get all tiff files
    tiff_files = list(input_path.glob("*.tif*"))

    if not tiff_files:
        click.echo("No TIFF files found in the input folder")
        return

    for tiff_file in tiff_files:
        # Read 16-bit image
        img = cv2.imread(str(tiff_file), cv2.IMREAD_ANYDEPTH)

        if img is None:
            click.echo(f"Failed to read {tiff_file}")
            continue

        # Clip values to specified range
        img = np.clip(img, min_value, max_value)

        # Normalize to 0-255 range
        img = ((img - min_value) / (max_value - min_value) * 255).astype(np.uint8)

        # Save as BMP
        output_file = output_path / f"{tiff_file.stem}.bmp"
        success = cv2.imwrite(str(output_file), img)

        if success:
            click.echo(f"Converted {tiff_file.name} to {output_file.name}")
        else:
            click.echo(f"Failed to convert {tiff_file.name}")


if __name__ == "__main__":
    convert_tiff_to_bmp()