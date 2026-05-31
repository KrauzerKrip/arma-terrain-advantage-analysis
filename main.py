import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. DEM / TERRAIN PROFILE EXTRACTION
# ==========================================

def _read_dem_header(dem_path):
    """Reads and parses the top 6 header lines of an ASC DEM file."""
    headers = {}
    with open(dem_path, 'r') as f:
        for _ in range(6):
            parts = f.readline().split()
            headers[parts[0]] = float(parts[1])
    return headers


def _extract_dem_crop(dem_path, headers, p_a, p_b, max_map_size):
    """Extracts a tightly bounded sub-grid array around the line segment AB to save memory."""
    cellsize = headers['cellsize']
    nodata = headers['NODATA_value']
    ncols = int(headers['ncols'])
    nrows = int(headers['nrows'])

    # Map coordinates to raw row/col indices (ignoring orientation for bounding box)
    c_min = max(0, int(min(p_a[0], p_b[0]) // cellsize) - 1)
    c_max = min(ncols, int(np.ceil(max(p_a[0], p_b[0]) / cellsize)) + 2)
    
    # Arma Y=0 is matrix row bottom, map inversely 
    r_start = max(0, int((max_map_size - max(p_a[1], p_b[1])) // cellsize) - 1)
    r_end = min(nrows, int(np.ceil((max_map_size - min(p_a[1], p_b[1])) / cellsize)) + 2)
    
    skiprows = 6 + r_start
    max_rows = max(1, r_end - r_start)
    
    dem_chunk = np.loadtxt(dem_path, skiprows=skiprows, max_rows=max_rows)
    dem_chunk = np.atleast_2d(dem_chunk)
    
    dem_crop = dem_chunk[:, c_min:c_max]
    dem_crop[dem_crop == nodata] = np.nan
    dem_crop[dem_crop < -5000] = np.nan
    
    return dem_crop, r_start, c_min


def _sample_profile_heights(dem_crop, headers, r_start, c_min, p_a, p_b, step_meters, max_map_size):
    """Interpolates coordinates and maps elevations linearly across the profile slice."""
    cellsize = headers['cellsize']
    
    crop_rows = np.arange(r_start, r_start + dem_crop.shape[0])
    crop_cols = np.arange(c_min, c_min + dem_crop.shape[1])
    
    crop_x = (crop_cols * cellsize) + (cellsize / 2)
    crop_y = max_map_size - ((crop_rows * cellsize) + (cellsize / 2))
    
    p_a, p_b = np.array(p_a), np.array(p_b)
    total_distance = np.linalg.norm(p_b - p_a)
    
    num_samples = int(np.ceil(total_distance / step_meters)) + 1
    sample_distances = np.linspace(0, total_distance, num_samples)
    
    direction = (p_b - p_a) / total_distance if total_distance > 0 else np.zeros(2)
    sample_coords = p_a + sample_distances[:, np.newaxis] * direction

    ascending_y = crop_y[::-1]
    corrected_matrix = np.flipud(dem_crop).T

    interp = RegularGridInterpolator(
        (crop_x, ascending_y), 
        corrected_matrix, 
        bounds_error=False, 
        fill_value=np.nan,
        method='linear'
    )

    profile = []
    for dist, (sx, sy, sz) in zip(sample_distances, sample_coords):
        height = float(interp([sx, sy])[0])
        profile.append({
            'distance_from_a': round(dist, 2),
            'x': round(sx, 2),
            'y': round(sy, 2),
            'height': round(height, 2) if not np.isnan(height) else None
        })
    return profile


def get_height_profile_linear(dem_path, p_a, p_b, step_meters=1.0, max_map_size=30720):
    """Returns a sequential list of coordinates and heights along a path from A to B."""
    headers = _read_dem_header(dem_path)
    dem_crop, r_start, c_min = _extract_dem_crop(dem_path, headers, p_a, p_b, max_map_size)
    return _sample_profile_heights(dem_crop, headers, r_start, c_min, p_a, p_b, step_meters, max_map_size)


# ==========================================
# 2. DATA PROCESSING & PARSING
# ==========================================

def _parse_kill_line(line):
    """Processes a single raw JSONL string from a kills log file."""
    kill = json.loads(line)
    if kill.get('killKind') == 'friendly':
        return None

    p1 = np.array([kill['killerPosition']['x'], kill['killerPosition']['y'], kill['killerPosition']['z']])
    p2 = np.array([kill['killedPosition']['x'], kill['killedPosition']['y'], kill['killedPosition']['z']])
    
    return {
        'engagement_dist': np.linalg.norm(p1[:2] - p2[:2]),
        'duel_delta_z': p1[2] - p2[2],
        'killer_position': p1,
        'killed_position': p2,
        'killer_side': kill['killerSide']
    }


def process_sandbox_data(data_dir):
    """Iterates through sandbox test runs to construct an evaluation pandas DataFrame."""
    dataset = []
    
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        manifest_path = os.path.join(folder_path, 'manifest.json')
        kills_path = os.path.join(folder_path, 'kills.jsonl')

        if not (os.path.exists(manifest_path) and os.path.exists(kills_path)):
            continue

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            macro_profile = manifest['candidate']['profile']    
            summary = manifest['summary']

            with open(kills_path, 'r') as f:
                for line in f:
                    kill_data = _parse_kill_line(line)
                    if not kill_data:
                        continue

                    # Merge summary metrics with situational kill records
                    kill_data.update({
                        'test_id': manifest['id'],
                        'test_duration_ms': summary['durationMs'],
                        'macro_height_delta': macro_profile['heightDelta'],
                        'macro_max_slope': macro_profile['maxSlopeDeg'],
                        'match_winner': 0 if summary['endReason'] == 'blufor_eliminated' else 1
                    })
                    if kill_data['killer_side'] == 'blufor':
                        # BLUFOR killed OPFOR -> BLUFOR height minus OPFOR height
                        blufor_delta_z = kill_data['killer_position'][2] - kill_data['killed_position'][2]
                    else:
                        # OPFOR killed BLUFOR -> BLUFOR height (victim) minus OPFOR height (killer)
                        blufor_delta_z = kill_data['killed_position'][2] - kill_data['killer_position'][2]

                    kill_data['blufor_delta_z'] = blufor_delta_z
                    dataset.append(kill_data)

        except KeyError as e:
            print(f"Couldn't process {folder}: {e}")
                
    return pd.DataFrame(dataset)


# ==========================================
# 3. METRICS AGGREGATION
# ==========================================

def compute_average_terrain_profile(all_line_series, num_bins=100, relative=False):
    """
    Computes the absolute or relative average terrain height profile across multiple lines.
    
    Args:
        all_line_series: List of profile lists (dicts containing distance and height).
        num_bins: Target resolution segments for percentage mapping.
        relative: If True, profile scales from delta starting at 0m elevation.
    """
    standardized_profiles = []
    target_percentages = np.linspace(0.0, 1.0, num_bins)
    
    for profile in all_line_series:
        clean_nodes = [node for node in profile if node['height'] is not None]
        if len(clean_nodes) < 2:
            continue
            
        distances = np.array([node['distance_from_a'] for node in clean_nodes])
        max_dist = distances[-1]
        if max_dist == 0:
            continue
            
        heights = np.array([node['height'] for node in clean_nodes])
        if relative:
            heights = heights - clean_nodes[0]['height']
            
        pct_distances = distances / max_dist
        resampled_heights = np.interp(target_percentages, pct_distances, heights)
        standardized_profiles.append(resampled_heights)
        
    if not standardized_profiles:
        raise ValueError("No valid profiles found to aggregate.")
        
    profile_matrix = np.array(standardized_profiles)
    
    return {
        'percent_steps': target_percentages * 100,
        'average_heights': np.mean(profile_matrix, axis=0),
        'min_heights': np.min(profile_matrix, axis=0),
        'max_heights': np.max(profile_matrix, axis=0)
    }


# ==========================================
# 4. PLOTTING & DATA VISUALIZATION
# ==========================================

def plot_height_profile(profile_data, output_filename):
    """Plots the terrain cross-section profile from Point A to Point B."""
    sanitized_data = [node for node in profile_data if node['height'] is not None]
    if not sanitized_data:
        print("Error: No valid elevation data found along this segment to plot.")
        return

    distances = np.array([node['distance_from_a'] for node in sanitized_data])
    heights = np.array([node['height'] for node in sanitized_data])
    
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    ax.plot(distances, heights, color='#1f77b4', linewidth=2.5, label='Terrain Elevation')
    
    min_y_fill = np.min(heights) - 5  
    ax.fill_between(distances, heights, min_y_fill, color='#1f77b4', alpha=0.15)
    
    ax.set_title('Terrain Elevation Profile (Point A $\\rightarrow$ Point B)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Distance from Point A (meters)', fontsize=11, labelpad=10)
    ax.set_ylabel('Elevation (meters)', fontsize=11, labelpad=10)
    
    ax.set_xlim(0, np.max(distances))
    ax.set_ylim(min_y_fill, np.max(heights) + 5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Profile plot successfully saved as: {output_filename}")


def plot_aggregated_terrain(avg_data, output_filename):
    """Plots the normalized average terrain profile with min/max variation bounds."""
    pct = avg_data['percent_steps']
    avg_h = avg_data['average_heights']
    min_h = avg_data['min_heights']
    max_h = avg_data['max_heights']
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.plot(pct, avg_h, color='#2ca02c', linewidth=3, label='Average Terrain Shape')
    ax.fill_between(pct, min_h, max_h, color='#2ca02c', alpha=0.1, label='Terrain Variation Envelope')
    
    ax.set_title('Normalized Dataset Terrain Profile', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('From killer to killed (%)', fontsize=11, labelpad=10)
    ax.set_ylabel('Elevation (meters)', fontsize=11, labelpad=10)
    
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()


def draw_jointplot(df, output_path):
    sns.jointplot(data=df, x='engagement_dist', y='duel_delta_z', kind='hex', cmap='Blues')
    plt.savefig(output_path)
    plt.close()
    print(f"Jointplot saved to: {output_path}")


def logi_regression(df, output_path):
    """Plots a true logistic regression curve using structural macro terrain deltas."""
    print("\n--- Generating Match Outcome Logistic Regression ---")
    
    # Compress the dataframe back to one unique row per match
    match_df = df.groupby('test_id').agg({
        'macro_height_delta': 'first',
        'match_winner': 'first'
    }).reset_index()

    plt.figure(figsize=(10, 6))
    
    # Plotting using the updated metrics
    ax = sns.regplot(
        data=match_df, 
        x='macro_height_delta', 
        y='match_winner', 
        logistic=True, 
        y_jitter=0.03,
        scatter_kws={'alpha': 0.5, 's': 40, 'color': '#1f77b4'},
        line_kws={'color': '#d62728', 'linewidth': 3}  # Striking red trendline
    )
    
    ax.set_title('Probability of BLUFOR Victory vs. Initial Elevation Delta', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('BLUFOR Height Advantage (meters, Negative = OPFOR High Ground)', fontsize=11)
    ax.set_ylabel('Probability of BLUFOR Win', fontsize=11)
    
    ax.set_xlim(match_df['macro_height_delta'].min() - 5, match_df['macro_height_delta'].max() + 5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Proper Logistic Regression curve saved to: {output_path}")


def make_classification(df, output_path):
    print("\n--- Training Random Forest Feature Importance ---")
    
    # Clean up any missing values that might break the classifier
    clean_df = df.dropna(subset=['macro_height_delta', 'macro_max_slope', 'engagement_dist', 'duel_delta_z'])
    
    if clean_df.empty:
        print("Error: No data available for classification.")
        return

    # Define features that mix macro layout with micro engagement mechanics
    X = clean_df[[
        'macro_height_delta', # Macro elevation advantage
        'macro_max_slope',          # Map steepness/choke potential
        'engagement_dist',          # Combat range (proxy for spacing/open ground)
        'duel_delta_z'              # Micro-elevation (who held the high ground in individual duels)
    ]]
    
    # Rename columns for a cleaner, human-readable plot
    X = X.rename(columns={
        'macro_height_delta': 'Macro Height Delta (BLUFOR)',
        'macro_max_slope': 'Max Map Slope (Degrees)',
        'engagement_dist': 'Engagement Distance',
        'duel_delta_z': 'Local Kill Height Delta (Z)'
    })
    
    y = clean_df['match_winner']

    # Train the Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Extract and format feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    # Use a nice teal color palette for presentation clarity
    importances.plot(kind='barh', color='#2ca02c', edgecolor='black', alpha=0.8)
    
    plt.title('What Predicts a Match Winner? (Feature Importance)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Relative Importance Score (0.0 to 1.0)', fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Classification feature importance chart saved to: {output_path}")


# ==========================================
# 5. EXECUTION ENTRY POINT
# ==========================================

def run_baseline_metrics_pipeline(df):
    """Calculates and prints simple baseline data stats."""
    df['distance'] = df.apply(
        lambda row: np.linalg.norm(row['killer_position'] - row['killed_position']), axis=1
    )
    print("\n--- Baseline Metrics ---")
    print(f"Mean Engagement Distance: {df['distance'].mean():.2f} meters")
    print(f"Standard Deviation: {df['distance'].std():.2f} meters")
    print(df.info())


def run_terrain_profile_pipeline(df, dem_path, output_path, limit=30):
    """Extracts, plots, and aggregates sequential spatial lines from points A to B."""
    print("\n--- Running Terrain Profile Aggregation Pipeline ---")
    first_positions = df[['killer_position', 'killed_position']].iloc[:limit]
    
    profiles = []
    for i, (killer, killed) in enumerate(first_positions.to_numpy()):
        # Extract individual metric profiles
        profile = get_height_profile_linear(dem_path, killer, killed)
        profiles.append(profile)
        
        # Optional: Uncomment if you want to output all individual line images 
        # plot_height_profile(profile, f".data/plots/profile_{i}.png")
    
    # Process relative mathematical adjustments and save the structural envelope
    average_terrain_profile = compute_average_terrain_profile(profiles, relative=True)
    plot_aggregated_terrain(average_terrain_profile, output_path)
    print(f"Aggregated terrain profile saved to: {output_path}")


def run_histogram_pipeline(df, output_path):
    """Generates and styles a distribution histogram of combat relative heights."""
    print("\n--- Generating Killer Advantage Histogram ---")
    plt.figure(figsize=(10, 6))
    ax = df['duel_delta_z'].plot.hist(bins=30, edgecolor='black', alpha=0.7)

    # Accentuate even elevation baseline
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Even Elevation')

    plt.title('Killer Elevation Advantage Delta Z)')
    plt.xlabel('Elevation Difference (Killer Height - Victim Height)')
    plt.ylabel('Number of Engagements')
    plt.legend()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved to: {output_path}")


def print_win_percentages(df):
    """Calculates and prints the clean win percentage for both BLUFOR and OPFOR."""
    print("\n--- Match Outcome Statistics ---")
    
    # Compress the dataframe so we have exactly one record per unique match
    match_df = df.groupby('test_id').agg({
        'match_winner': 'first'
    }).reset_index()
    
    total_matches = len(match_df)
    if total_matches == 0:
        print("No matches found to calculate win rates.")
        return
        
    # Calculate wins
    blufor_wins = match_df['match_winner'].sum()  # Counts all the 1s
    opfor_wins = total_matches - blufor_wins      # The remaining 0s
    
    # Turn into percentages
    blufor_pct = (blufor_wins / total_matches) * 100
    opfor_pct = (opfor_wins / total_matches) * 100
    
    print(f"Total Simulated Matches: {total_matches}")
    print(f"BLUFOR (Attacker) Wins : {blufor_wins} ({blufor_pct:.1f}%)")
    print(f"OPFOR  (Defender) Wins : {opfor_wins} ({opfor_pct:.1f}%)")


def main():
    # Core Data Assembly Line (Required)
    df = process_sandbox_data(".data/terrain_advantage_data")
    if df.empty:
        print("Dataframe is empty. Verify data directory contents.")
        return
    print(df)

    # -----------------------------------------------------------------
    # PIPELINE SWITCHES
    # Comment or uncomment any single line below to toggle features off/on
    # -----------------------------------------------------------------
    run_baseline_metrics_pipeline(df)
    run_terrain_profile_pipeline(df, dem_path=".data/altis/dem.asc", output_path=".data/average_terrain_profile.png", limit=30)
    run_histogram_pipeline(df, ".data/delta_z_histogram.png")
    draw_jointplot(df, ".data/jointplot.png")
    logi_regression(df, ".data/lmplot.png")
    make_classification(df, ".data/classification.png")
    print_win_percentages(df)


if __name__ == "__main__":
    main()