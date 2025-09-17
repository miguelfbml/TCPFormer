import numpy as np

def convert_yolo_npz_to_original(yolo_npz_path, output_npz_path):
    data = np.load(yolo_npz_path, allow_pickle=True)['data'].item()
    new_data = {}
    for key in data:
        # Each value is a list of length 1, containing a dict
        cam0_data = data[key][0]
        # Wrap in a dict with camera key '0'
        new_data[key] = [{'0': cam0_data}]
    # Save to new npz file
    np.savez_compressed(output_npz_path, data=new_data)
    print(f"✓ Converted file saved to: {output_npz_path}")

if __name__ == "__main__":
    convert_yolo_npz_to_original("data_train_3dhp_yolo.npz", "data_train_3dhp_yolo_fixed.npz")