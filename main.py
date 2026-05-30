import os
import json
import pandas as pd
import numpy as np

import numpy as np
import os

def get_height_profile_linear(dem_path, p_a, p_b, step_meters=1.0, max_map_size=30720):
    """
    Returns a sequential list of coordinates and heights along a path from A to B
    sampled at a fixed metric interval (e.g., every 1 meter).
    
    Returns:
    - list of dicts: [{'x': x, 'y': y, 'distance': dist, 'height': h}, ...]
    """
    # 1. Read Header
    headers = {}
    with open(dem_path, 'r') as f:
        for _ in range(6):
            parts = f.readline().split()
            headers[parts[0]] = float(parts[1])
            
    cellsize = headers['cellsize']
    nodata = headers['NODATA_value']
    ncols = int(headers['ncols'])
    nrows = int(headers['nrows'])

    # 2. Extract a tightly bounded chunk around the line segment to save RAM
    # Map coordinates to raw row/col indices (ignoring orientation for bounding box)
    c_min = max(0, int(min(p_a[0], p_b[0]) // cellsize) - 1)
    c_max = min(ncols, int(np.ceil(max(p_a[0], p_b[0]) / cellsize)) + 2)
    
    # Remember: Arma Y=0 is matrix row bottom, so we map inversely 
    r_start = max(0, int((max_map_size - max(p_a[1], p_b[1])) // cellsize) - 1)
    r_end = min(nrows, int(np.ceil((max_map_size - min(p_a[1], p_b[1])) / cellsize)) + 2)
    
    skiprows = 6 + r_start
    max_rows = max(1, r_end - r_start)
    
    # Load just the sub-grid containing our line segment
    dem_chunk = np.loadtxt(dem_path, skiprows=skiprows, max_rows=max_rows)
    dem_chunk = np.atleast_2d(dem_chunk)
    
    # Slice columns out of the chunk
    dem_crop = dem_chunk[:, c_min:c_max]
    dem_crop[dem_crop == nodata] = np.nan
    dem_crop[dem_crop < -5000] = np.nan

    # 3. Create coordinate grids matching the cropped area
    # Compute real Arma X and Y centers for every pixel in our cropped array
    crop_rows = np.arange(r_start, r_start + dem_crop.shape[0])
    crop_cols = np.arange(c_min, c_min + dem_crop.shape[1])
    
    crop_x = (crop_cols * cellsize) + (cellsize / 2)
    crop_y = max_map_size - ((crop_rows * cellsize) + (cellsize / 2))
    
    # 4. Generate sequential profile sampling points from A to B
    p_a = np.array(p_a)
    p_b = np.array(p_b)
    total_distance = np.linalg.norm(p_b - p_a)
    
    # Create distances tracking from 0m (A) to total_distance (B)
    num_samples = int(np.ceil(total_distance / step_meters)) + 1
    sample_distances = np.linspace(0, total_distance, num_samples)
    
    # Compute actual (X,Y) positions along the vector line segment
    # Direction vector * step + starting point
    direction = (p_b - p_a) / total_distance if total_distance > 0 else np.zeros(2)
    sample_coords = p_a + sample_distances[:, np.newaxis] * direction

    # 5. Build 2D Interpolator using scipy for smooth lookup
    from scipy.interpolate import RegularGridInterpolator
    
    # Sort crop_y so it is strictly ascending for the interpolator
    ascending_y = crop_y[::-1]

    # Correct matrix alignment:
    # We need the matrix rows to match ascending_y. 
    # Since dem_crop originally has rows going DOWN, we flip the rows vertically.
    corrected_matrix = np.flipud(dem_crop).T

    # RegularGridInterpolator expects strictly ascending coordinates.
    # Because crop_y goes from top (high value) to bottom (low value), we invert it.
    interp = RegularGridInterpolator(
        (crop_x, ascending_y), 
        corrected_matrix, 
        bounds_error=False, 
        fill_value=np.nan,
        method='linear'          # Bilinear interpolation fills smooth values between cell edges
    )

    # 6. Gather the final sequenced array profile
    profile = []
    for dist, (sx, sy, sz) in zip(sample_distances, sample_coords):
        # Sample the exact elevation at this metric point
        height = float(interp([sx, sy])[0])
        
        profile.append({
            'distance_from_a': round(dist, 2),
            'x': round(sx, 2),
            'y': round(sy, 2),
            'height': round(height, 2) if not np.isnan(height) else None
        })
        
    return profile

def process_sandbox_data(data_dir):
    dataset = []
    
    # Iterate through all test run folders
    for folder in os.listdir(data_dir):
        try:
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            manifest_path = os.path.join(folder_path, 'manifest.json')
            kills_path = os.path.join(folder_path, 'kills.jsonl')

            if not (os.path.exists(manifest_path) and os.path.exists(kills_path)):
                continue

            # Load macro metrics
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            macro_profile = manifest['candidate']['profile']
            summary = manifest['summary']

            # Process individual duels
            with open(kills_path, 'r') as f:
                for line in f:
                    kill = json.loads(line)

                    # Filter out friendly fire incidents
                    if kill.get('killKind') == 'friendly':
                        continue

                    # Calculate exact spatial values
                    p1 = np.array([kill['killerPosition']['x'], kill['killerPosition']['y'], kill['killerPosition']['z']])
                    p2 = np.array([kill['killedPosition']['x'], kill['killedPosition']['y'], kill['killedPosition']['z']])
                    dist_2d = np.linalg.norm(p1 - p2)

                    z_killer = kill['killerPosition']['z']
                    z_killed = kill['killedPosition']['z']
                    delta_z = z_killer - z_killed

                    dataset.append({
                        'test_id': manifest['id'],
                        'test_duration_ms': summary['durationMs'],
                        'macro_height_delta': macro_profile['heightDelta'],
                        'macro_max_slope': macro_profile['maxSlopeDeg'],
                        'engagement_dist': dist_2d,
                        'duel_delta_z': delta_z,
                        'killer_position': p1,
                        'killed_position': p2,
                        'killer_side': kill['killerSide'], # Target variable: Who won the duel?
                        'match_winner': 1 if summary['endReason'] == 'blufor_eliminated' else 0 # 1 = Defense Won, 0 = Offense Won
                    })
        except KeyError as e:
            print(f"Couldn't process {folder}: {e}")
                
    return pd.DataFrame(dataset)

# Usage
# df = process_sandbox_data('./akela/.data/terrain_advantage_data')
import matplotlib.pyplot as plt
import numpy as np

def plot_height_profile(profile_data, output_filename="terrain_profile.png"):
    """
    Plots the terrain cross-section profile from Point A to Point B.
    Handles missing or None elevation entries gracefully.
    """
    # 1. Extract and sanitize distances and heights dynamically
    # We skip nodes where height is None to keep NumPy and Matplotlib happy
    sanitized_data = [node for node in profile_data if node['height'] is not None]
    
    if not sanitized_data:
        print("Error: No valid elevation data found along this segment to plot.")
        return

    distances = np.array([node['distance_from_a'] for node in sanitized_data])
    heights = np.array([node['height'] for node in sanitized_data])
    
    # 2. Initialize the plot layout
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    
    # 3. Plot the elevation line
    ax.plot(distances, heights, color='#1f77b4', linewidth=2.5, label='Terrain Elevation')
    
    # Establish a safe floor for the visual fill
    min_y_fill = np.min(heights) - 5  
    
    # 4. Fill the area underneath safely (no Nones!)
    ax.fill_between(distances, heights, min_y_fill, color='#1f77b4', alpha=0.15)
    
    # 5. Styling, labels, and titles
    ax.set_title('Terrain Elevation Profile (Point A $\\rightarrow$ Point B)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Distance from Point A (meters)', fontsize=11, labelpad=10)
    ax.set_ylabel('Elevation (meters)', fontsize=11, labelpad=10)
    
    # Set boundaries tightly around active metrics
    ax.set_xlim(0, np.max(distances))
    ax.set_ylim(min_y_fill, np.max(heights) + 5)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    # 6. Optimize layout spacing and save the file
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    print(f"Profile plot successfully saved as: {output_filename}")

def main():
    df = process_sandbox_data(".data/terrain_advantage_data")
    first_ten_positions = df[['killer_position', 'killed_position']].iloc[:10]
    df['distance'] = df.apply(lambda row: np.linalg.norm(np.array(row['killer_position']) - np.array(row['killed_position'])), axis=1)
    print(f"Mean: {df['distance'].mean()}")
    print(f"Standard deviation: {df['distance'].std()}")
    i = 0
    for killer, killed in first_ten_positions.to_numpy():
        profile = get_height_profile_linear(".data/altis/dem.asc", killer, killed)
        plot_height_profile(profile, f".data/plots/profile_{i}")
        i = i + 1



if __name__ == "__main__":
    main()
