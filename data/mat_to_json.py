import scipy.io
import json
import numpy as np

def mat_to_json(mat_file_path, json_file_path):
    """
    Convert a .mat file to a JSON file.
    
    Parameters:
    - mat_file_path: Path to the input .mat file.
    - json_file_path: Path to the output JSON file.
    """
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)

    # Remove MATLAB-specific metadata keys
    data = {key: value for key, value in mat_data.items() if not key.startswith("__")}

    # Convert numpy arrays to lists for JSON serialization
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    serialized_data = {key: serialize(value) for key, value in data.items()}

    # Write to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(serialized_data, json_file, indent=4)
    
    print(f"Converted {mat_file_path} to {json_file_path}")

# Example usage
mat_file_path = "/Users/noellelaw/Downloads/geomat_scene_rh_labels.mat"  # Path to your .mat file
json_file_path = "data/geomat_scene_rh_labels.json"  # Path to the desired JSON file
mat_to_json(mat_file_path, json_file_path)