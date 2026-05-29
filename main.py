import os
import json
import pandas as pd
import numpy as np

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
                    p1 = np.array([kill['killerPosition']['x'], kill['killerPosition']['y']])
                    p2 = np.array([kill['killedPosition']['x'], kill['killedPosition']['y']])
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


def main():
    data = process_sandbox_data("/home/krauzerkrip/projects/ml/terrain-akela/.data/terrain_advantage_data")
    print(data.isna().sum())



if __name__ == "__main__":
    main()
