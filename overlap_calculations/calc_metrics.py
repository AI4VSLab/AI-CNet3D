import os
import cv2
import numpy as np
from pathlib import Path
import json
import pandas as pd


def calculate_intersection_metrics(care_heatmap, mask, mask_name="mask"):
    """
    Calculate various intersection metrics between CARE heatmap and binary mask

    Args:
        care_heatmap (np.array): CARE intensity values (0-1)
        mask (np.array): Binary mask (0 or 1)
        mask_name (str): Name for reporting

    Returns:
        dict: Dictionary of intersection metrics
    """
    # Ensure inputs are proper format
    care_heatmap = care_heatmap.astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)  # Ensure binary
    # care_median = np.median(care_heatmap[care_heatmap > 0])
    care_median = np.percentile(care_heatmap[care_heatmap > 0], 75)

    care_binary = (care_heatmap > care_median).astype(np.float32)

    intersection = np.sum(mask * care_binary)

    mask_area = np.sum(mask)
    # Basic counts
    mask_area = np.sum(mask)
    total_pixels = mask.size
    care_area = np.sum(care_binary)

    metrics = {}

    metrics['precision'] = intersection / mask_area if mask_area > 0 else 0.0
    metrics['recall'] = intersection / care_area if care_area > 0 else 0.0
    metrics['intersection'] = intersection

    care_in_mask = care_heatmap * mask
    metrics['mean_care_in_mask'] = np.sum(
        care_in_mask) / mask_area if mask_area > 0 else 0.0
    metrics['total_care_in_mask'] = np.sum(care_in_mask)
    metrics['max_care_in_mask'] = np.max(
        care_in_mask) if mask_area > 0 else 0.0

    metrics['mask_coverage'] = np.sum((care_in_mask > care_median).astype(
        np.float32)) / mask_area if mask_area > 0 else 0.0

    metrics['mask_in_care'] = np.sum(mask*care_binary)/np.sum(care_binary)

    # 8. Relative CARE enrichment in mask vs outside
    care_outside_mask = care_heatmap * (1 - mask)
    outside_area = total_pixels - mask_area
    if outside_area > 0:
        mean_care_outside = np.sum(care_outside_mask) / outside_area
        if mean_care_outside > 0:
            metrics['care_enrichment_ratio'] = float(
                metrics['mean_care_in_mask'] / mean_care_outside)
        else:
            metrics['care_enrichment_ratio'] = float(
                'inf') if metrics['mean_care_in_mask'] > 0 else 1.0
    else:
        metrics['care_enrichment_ratio'] = 1.0

    return metrics


def process_group_folder(group_path):
    """
    Process a single group folder to subtract masks from care_raw.png and calculate metrics

    Args:
        group_path (str): Path to the group folder

    Returns:
        dict: Results including processed image and metrics
    """
    group_path = Path(group_path)
    print(f"Processing {group_path.name}...")

    # Find care_raw.png
    care_raw_path = group_path / "care_raw.png"
    if not care_raw_path.exists():
        print(f"  Warning: care_raw.png not found in {group_path.name}")
        return None

    # Load care_raw as grayscale
    care_raw = cv2.imread(str(care_raw_path), cv2.IMREAD_GRAYSCALE)
    if care_raw is None:
        print(f"  Error: Could not load care_raw.png from {group_path.name}")
        return None

    # Convert to float for processing (0-1 range)
    care_raw = care_raw.astype(np.float32) / 255.0

    # Find all masks
    gcl_masks = list(group_path.glob("gcl_mask*.png"))
    rnfl_masks = list(group_path.glob("rnfl_mask*.png"))
    all_masks = gcl_masks + rnfl_masks

    print(
        f"  Found {len(gcl_masks)} GCL mask(s) and {len(rnfl_masks)} RNFL mask(s)")

    # Calculate metrics BEFORE subtraction (original intersection)
    all_metrics = {}

    # Process GCL masks
    for i, mask_path in enumerate(gcl_masks):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = (mask > 127).astype(np.float32)
            mask_name = f"gcl_mask_{i}" if len(gcl_masks) > 1 else "gcl_mask"

            metrics = calculate_intersection_metrics(care_raw, mask, mask_name)
            if metrics:
                all_metrics[mask_name] = metrics
                print(f"  {mask_name} metrics: {metrics}")
        else:
            print(f"  Warning: Could not load {mask_path.name}")

    # Process RNFL masks
    for i, mask_path in enumerate(rnfl_masks):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = (mask > 127).astype(np.float32)
            mask_name = f"rnfl_mask_{i}" if len(
                rnfl_masks) > 1 else "rnfl_mask"

            metrics = calculate_intersection_metrics(care_raw, mask, mask_name)
            if metrics:
                all_metrics[mask_name] = metrics
                print(f"  {mask_name} metrics: {metrics}")
        else:
            print(f"  Warning: Could not load {mask_path.name}")

    # Now perform subtraction to create processed image
    result = care_raw.copy()

    # Subtract all masks
    for mask_path in all_masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Convert mask to binary (0 or 1)
            mask = (mask > 127).astype(np.float32)
            # Subtract mask from result
            result = result - mask
            print(f"  Subtracted {mask_path.name}")

    # Clip values to ensure they stay in valid range
    result = np.clip(result, 0.0, 1.0)

    # Convert back to 8-bit for saving
    result_8bit = (result * 255).astype(np.uint8)

    if os.path.exists(str(group_path / "image.png")):
        heatmap = cv2.applyColorMap(result_8bit, cv2.COLORMAP_HOT)
        original_oct = cv2.imread(str(group_path / "image.png"))
        print(original_oct.shape, heatmap.shape)
        combined = cv2.addWeighted(original_oct, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(str(group_path / "image_processed.png"), combined)

    # Save the processed image
    output_path = group_path / "care_processed.png"
    cv2.imwrite(str(output_path), result_8bit)
    print(f"  Saved processed image to {output_path.name}")

    # Save metrics as JSON (convert numpy types to Python types)
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif obj is np.inf:
            return "infinity"
        elif obj is -np.inf:
            return "-infinity"
        elif np.isnan(obj):
            return "NaN"
        else:
            return obj

    metrics_path = group_path / "intersection_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(convert_numpy_types(all_metrics), f, indent=2)
    print(f"  Saved metrics to {metrics_path.name}")

    return {
        'processed_image': result,
        'original_image': care_raw,
        'metrics': all_metrics,
        'group_path': str(group_path)
    }


def process_all_groups(base_path, save_summary=True):
    """
    Process all group folders in the base path, handling full_X/group_Y structure

    Args:
        base_path (str): Path to the matched folder containing full_X folders
        save_summary (bool): Whether to save a summary CSV of all metrics

    Returns:
        dict: Nested dictionary of all results
    """
    base_path = Path(base_path)

    # Find all full_X folders
    full_folders = [f for f in base_path.iterdir() if f.is_dir()
                    and f.name.startswith('full_')]

    if not full_folders:
        print("No full_X folders found!")
        return

    print(f"Found {len(full_folders)} full folder(s)")

    all_results = {}
    summary_data = []
    total_groups_processed = 0

    for full_folder in sorted(full_folders):
        print(f"\n=== Processing {full_folder.name} ===")

        # Find all group folders within this full folder
        group_folders = [f for f in full_folder.iterdir(
        ) if f.is_dir() and f.name.startswith('group_')]

        if not group_folders:
            print(f"  No group folders found in {full_folder.name}")
            continue

        print(
            f"  Found {len(group_folders)} group folder(s) in {full_folder.name}")

        full_results = {}
        for group_folder in sorted(group_folders):
            result = process_group_folder(group_folder)
            if result is not None:
                full_results[group_folder.name] = result
                total_groups_processed += 1

                # Add to summary data
                for mask_name, metrics in result['metrics'].items():
                    summary_row = {
                        'full_folder': full_folder.name,
                        'group_folder': group_folder.name,
                        'mask_name': mask_name,
                        **metrics  # Unpack all metrics
                    }
                    summary_data.append(summary_row)

        all_results[full_folder.name] = full_results

    print(f"\n=== Processing complete! ===")
    print(
        f"Processed {total_groups_processed} groups across {len(full_folders)} full folders.")

    # Save summary CSV
    if save_summary and summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = base_path / "care_intersection_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary metrics to {summary_path}")

        # Print comprehensive summary statistics
        print(f"\n=== COMPREHENSIVE SUMMARY STATISTICS ===")
        print(f"Total masks processed: {len(summary_data)}")
        print(
            f"GCL masks: {len([x for x in summary_data if 'gcl' in x['mask_name']])}")
        print(
            f"RNFL masks: {len([x for x in summary_data if 'rnfl' in x['mask_name']])}")

        print(f"\n--- KEY INTERSECTION METRICS (AVERAGES) ---")
        key_metrics = [
            'precision', 'recall', 'intersection', 'care_enrichment_ratio', 'mask_in_care'
        ]

        for metric in key_metrics:
            if metric in summary_df.columns:
                mean_val = summary_df[metric].mean()
                std_val = summary_df[metric].std()
                print(f"{metric:30}: {mean_val:8.4f} ± {std_val:6.4f}")

        print(f"\n--- ALL METRICS (AVERAGES) ---")
        metric_columns = [col for col in summary_df.columns if col not in [
            'full_folder', 'group_folder', 'mask_name']]

        for metric in sorted(metric_columns):
            # Handle potential infinity values
            finite_values = summary_df[metric].replace(
                [np.inf, -np.inf], np.nan).dropna()
            if len(finite_values) > 0:
                mean_val = finite_values.mean()
                std_val = finite_values.std()
                min_val = finite_values.min()
                max_val = finite_values.max()
                print(
                    f"{metric:35}: {mean_val:8.4f} ± {std_val:6.4f} (range: {min_val:6.4f} - {max_val:6.4f})")
            else:
                print(f"{metric:35}: No finite values")

        # Print breakdown by mask type
        print(f"\n--- BREAKDOWN BY MASK TYPE ---")
        for mask_type in ['gcl', 'rnfl']:
            mask_data = summary_df[summary_df['mask_name'].str.contains(
                mask_type)]
            if len(mask_data) > 0:
                print(f"\n{mask_type.upper()} masks ({len(mask_data)} total):")
                for metric in key_metrics:
                    if metric in mask_data.columns:
                        finite_values = mask_data[metric].replace(
                            [np.inf, -np.inf], np.nan).dropna()
                        if len(finite_values) > 0:
                            mean_val = finite_values.mean()
                            print(f"  {metric:30}: {mean_val:8.4f}")
    enriched_masks = summary_df[summary_df['care_enrichment_ratio'] > 1.0]
    depleted_masks = summary_df[summary_df['care_enrichment_ratio'] < 1.0]

    print(
        f"Masks with CARE enrichment (ratio > 1.0): {len(enriched_masks)}/{len(summary_data)} ({100*len(enriched_masks)/len(summary_data):.1f}%)")
    print(
        f"Masks with CARE depletion (ratio < 1.0):  {len(depleted_masks)}/{len(summary_data)} ({100*len(depleted_masks)/len(summary_data):.1f}%)")

    return all_results


# Example usage
if __name__ == "__main__":
    print("CARE Processing and Intersection Metrics Calculator")

    # Update this path to point to your matched folder
    base_path = "/path/to/your/matched/data/"

    # Process all full_X/group_Y folders
    results = process_all_groups(base_path, save_summary=True)
