import numpy as np

def convert_test_to_train_format(test_npz_path, output_npz_path):
    # Load test set
    test_data = np.load(test_npz_path, allow_pickle=True)['data'].item()
    train_data = {}

    for seq in test_data:
        seq_dict = test_data[seq]
        # Wrap each test sequence as a list with one camera dict ('0')
        cam0_dict = {
            'data_3d': seq_dict['data_3d'],
            'data_2d': seq_dict['data_2d'],
            'valid': seq_dict['valid'],
            'camera': None
        }
        train_data[seq] = [{'0': cam0_dict}]

    np.savez_compressed(output_npz_path, data=train_data)
    print(f"✓ Converted test set saved as train format to: {output_npz_path}")

if __name__ == "__main__":
    convert_test_to_train_format("data_test_3dhp.npz", "data_train_3dhp_from_test.npz")