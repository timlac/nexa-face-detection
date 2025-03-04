import os
import pickle
import json
import numpy as np


def custom_serializer(obj):
    """
    Custom serializer to handle non-serializable objects.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif hasattr(obj, '__dict__'):
        return obj.__dict__  # Serialize objects with attributes
    else:
        return str(obj)  # Fallback to string representation


def convert_pickles_to_json(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.pckl'):  # Process only pickle files
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.json")

            try:
                # Load pickle file
                with open(input_path, 'rb') as pkl_file:
                    data = pickle.load(pkl_file)

                # Save as JSON
                with open(output_path, 'w') as json_file:
                    json.dump(data, json_file, default=custom_serializer, indent=4)

                print(f"Converted {file_name} to JSON.")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

def main():
    # Input and output folders
    inp = "../data/out/test_snippet_timestamps_2"
    outp = "../data/out/test_snippet_timestamps_2"

    convert_pickles_to_json(inp, outp)

if __name__ == '__main__':
    main()