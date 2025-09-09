import os
import numpy as np
from PIL import Image
from pathlib import Path
import cv2


def create_mask_overlay(image_path, gcl_masks, rnfl_masks, output_path,
                        gcl_color=(0, 255, 0), rnfl_color=(255, 0, 0), alpha=0.5):
    """
    Create an overlay image with GCL and RNFL masks applied in different colors.

    Args:
        image_path (str): Path to the original image
        gcl_masks (list): List of paths to GCL mask files
        rnfl_masks (list): List of paths to RNFL mask files
        output_path (str): Path where overlay image will be saved
        gcl_color (tuple): RGB color for GCL masks (default: green)
        rnfl_color (tuple): RGB color for RNFL masks (default: red)
        alpha (float): Transparency of overlay (0.0 = transparent, 1.0 = opaque)
    """

    # Load the original image
    original = Image.open(image_path).convert('RGB')
    original_array = np.array(original)

    # Create overlay array (same size as original)
    overlay = np.zeros_like(original_array)

    # Process GCL masks
    for gcl_mask_path in gcl_masks:
        if Path(gcl_mask_path).exists():
            # Load mask (should be grayscale)
            mask = Image.open(gcl_mask_path).convert('L')
            mask_array = np.array(mask)

            # Create binary mask (assuming white = mask, black = background)
            binary_mask = mask_array > 128  # Threshold at middle gray

            # Apply GCL color where mask is True
            overlay[binary_mask] = gcl_color

    # Process RNFL masks
    for rnfl_mask_path in rnfl_masks:
        if Path(rnfl_mask_path).exists():
            # Load mask (should be grayscale)
            mask = Image.open(rnfl_mask_path).convert('L')
            mask_array = np.array(mask)

            # Create binary mask
            binary_mask = mask_array > 128  # Threshold at middle gray

            # Apply RNFL color where mask is True
            overlay[binary_mask] = rnfl_color

    # Blend original image with overlay
    # Only blend where overlay is not black (where masks exist)
    mask_exists = np.any(overlay > 0, axis=2)

    result = original_array.copy().astype(np.float32)
    overlay_float = overlay.astype(np.float32)

    # Apply blending only where masks exist
    result[mask_exists] = (1 - alpha) * result[mask_exists] + \
        alpha * overlay_float[mask_exists]

    # Convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Save the result
    result_image = Image.fromarray(result)
    result_image.save(output_path)

    return result_image


def process_organized_dataset(dataset_root, output_suffix="_overlay",
                              gcl_color=(0, 255, 0), rnfl_color=(255, 0, 0), alpha=0.5):
    """
    Process the entire organized dataset to create overlay images.

    Args:
        dataset_root (str): Root directory of organized dataset
        output_suffix (str): Suffix to add to output filenames
        gcl_color (tuple): RGB color for GCL masks
        rnfl_color (tuple): RGB color for RNFL masks
        alpha (float): Transparency of overlay
    """

    dataset_root = Path(dataset_root)
    processed_count = 0
    error_count = 0

    # Process each full_x directory
    for full_dir in dataset_root.glob('full_*'):
        if not full_dir.is_dir():
            continue

        print(f"Processing {full_dir.name}...")

        # Process each group_x directory
        for group_dir in full_dir.glob('group_*'):
            if not group_dir.is_dir():
                continue

            try:
                # Find the main image
                image_path = group_dir / 'image.png'
                if not image_path.exists():
                    print(f"  Warning: No image.png found in {group_dir}")
                    continue

                # Find all GCL masks
                gcl_masks = list(group_dir.glob('gcl_mask*.png'))

                # Find all RNFL masks
                rnfl_masks = list(group_dir.glob('rnfl_mask*.png'))

                if not gcl_masks and not rnfl_masks:
                    print(f"  Warning: No masks found in {group_dir}")
                    continue

                # Create output filename
                output_filename = f"image{output_suffix}.png"
                output_path = group_dir / output_filename

                # Create the overlay
                create_mask_overlay(
                    image_path=str(image_path),
                    gcl_masks=[str(mask) for mask in gcl_masks],
                    rnfl_masks=[str(mask) for mask in rnfl_masks],
                    output_path=str(output_path),
                    gcl_color=gcl_color,
                    rnfl_color=rnfl_color,
                    alpha=alpha
                )

                processed_count += 1
                print(f"  ✓ Created overlay for {group_dir.name}")

            except Exception as e:
                error_count += 1
                print(f"  ✗ Error processing {group_dir.name}: {e}")

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} groups")
    print(f"Errors: {error_count}")

    return processed_count, error_count


def create_overlay_with_custom_colors(dataset_root, color_scheme="default"):
    """
    Create overlays with predefined color schemes.

    Args:
        dataset_root (str): Root directory of organized dataset
        color_scheme (str): Color scheme to use
    """

    color_schemes = {
        "default": {
            "gcl_color": (0, 255, 0),      # Green
            "rnfl_color": (255, 0, 0),     # Red
            "alpha": 0.5
        },
        "medical": {
            "gcl_color": (0, 255, 255),    # Cyan
            "rnfl_color": (255, 255, 0),   # Yellow
            "alpha": 0.6
        },
        "high_contrast": {
            "gcl_color": (255, 0, 255),    # Magenta
            "rnfl_color": (0, 255, 0),     # Green
            "alpha": 0.7
        },
        "subtle": {
            "gcl_color": (100, 200, 100),  # Light green
            "rnfl_color": (200, 100, 100),  # Light red
            "alpha": 0.3
        }
    }

    if color_scheme not in color_schemes:
        print(
            f"Unknown color scheme '{color_scheme}'. Available: {list(color_schemes.keys())}")
        return

    scheme = color_schemes[color_scheme]
    suffix = f"_overlay_{color_scheme}"

    print(f"Creating overlays with '{color_scheme}' color scheme...")
    print(f"GCL color: {scheme['gcl_color']}")
    print(f"RNFL color: {scheme['rnfl_color']}")
    print(f"Alpha: {scheme['alpha']}")

    return process_organized_dataset(
        dataset_root=dataset_root,
        output_suffix=suffix,
        gcl_color=scheme['gcl_color'],
        rnfl_color=scheme['rnfl_color'],
        alpha=scheme['alpha']
    )


def create_individual_overlays(dataset_root):
    """
    Create separate overlay images for each mask type.
    """

    dataset_root = Path(dataset_root)

    for full_dir in dataset_root.glob('full_*'):
        if not full_dir.is_dir():
            continue

        for group_dir in full_dir.glob('group_*'):
            if not group_dir.is_dir():
                continue

            try:
                image_path = group_dir / 'image.png'
                if not image_path.exists():
                    continue

                # Create GCL-only overlay
                gcl_masks = list(group_dir.glob('gcl_mask*.png'))
                if gcl_masks:
                    gcl_output = group_dir / "image_gcl_overlay.png"
                    create_mask_overlay(
                        image_path=str(image_path),
                        gcl_masks=[str(mask) for mask in gcl_masks],
                        rnfl_masks=[],  # No RNFL masks
                        output_path=str(gcl_output),
                        gcl_color=(0, 255, 0),
                        rnfl_color=(255, 0, 0),
                        alpha=0.5
                    )

                # Create RNFL-only overlay
                rnfl_masks = list(group_dir.glob('rnfl_mask*.png'))
                if rnfl_masks:
                    rnfl_output = group_dir / "image_rnfl_overlay.png"
                    create_mask_overlay(
                        image_path=str(image_path),
                        gcl_masks=[],  # No GCL masks
                        rnfl_masks=[str(mask) for mask in rnfl_masks],
                        output_path=str(rnfl_output),
                        gcl_color=(0, 255, 0),
                        rnfl_color=(255, 0, 0),
                        alpha=0.5
                    )

                print(f"Created individual overlays for {group_dir.name}")

            except Exception as e:
                print(
                    f"Error creating individual overlays for {group_dir.name}: {e}")


# Example usage
if __name__ == "__main__":
    # Set your dataset root path
    dataset_root = "/path/to/your/matched/data"

    print("Creating mask overlays...")
    print("GCL masks will be colored GREEN")
    print("RNFL masks will be colored BLUE")

    create_overlay_with_custom_colors(dataset_root, "medical")
