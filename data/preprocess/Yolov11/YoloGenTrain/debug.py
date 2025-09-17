import numpy as np

def debug_npz_structure(npz_path):
    data = np.load(npz_path, allow_pickle=True)['data'].item()
    print(f"Top-level keys: {list(data.keys())}")
    for key in data:
        print(f"\nKey: {key}")
        print(f"  Type: {type(data[key])}")
        if isinstance(data[key], list):
            print(f"  List length: {len(data[key])}")
            if len(data[key]) > 0:
                print(f"  First element type: {type(data[key][0])}")
                if isinstance(data[key][0], dict):
                    print(f"  Dict keys: {list(data[key][0].keys())}")
                    for k in data[key][0]:
                        print(f"    {k}: type={type(data[key][0][k])}, shape={getattr(data[key][0][k], 'shape', 'N/A')}")
        elif isinstance(data[key], dict):
            print(f"  Dict keys: {list(data[key].keys())}")
            for k in data[key]:
                print(f"    {k}: type={type(data[key][k])}, shape={getattr(data[key][k], 'shape', 'N/A')}")
        else:
            print(f"  Value: {data[key]}")

if __name__ == "__main__":
    debug_npz_structure("data_train_3dhp.npz")