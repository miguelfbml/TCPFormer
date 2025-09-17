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





def debug_one_entry(npz_path, entry_key=None):
    data = np.load(npz_path, allow_pickle=True)['data'].item()
    keys = list(data.keys())
    print(f"Top-level keys: {keys}")
    key = entry_key if entry_key is not None else keys[0]
    print(f"\nInspecting key: {key}")
    entry = data[key]
    print(f"  Type: {type(entry)}")
    if isinstance(entry, list):
        print(f"  List length: {len(entry)}")
        if len(entry) > 0 and isinstance(entry[0], dict):
            print(f"  Dict keys: {list(entry[0].keys())}")
            # Inspect camera '0'
            cam0 = entry[0].get('0')
            if cam0 is not None:
                print(f"\n  --- Inside camera '0' ---")
                print(f"    Keys: {list(cam0.keys())}")
                for k in cam0:
                    val = cam0[k]
                    print(f"    {k}: type={type(val)}, shape={getattr(val, 'shape', 'N/A')}")
                    if hasattr(val, 'shape') and hasattr(val, 'flatten'):
                        print(f"      Sample: {val.flatten()[:10]}")
            else:
                print("    Camera '0' not found in entry[0].")
    elif isinstance(entry, dict):
        print(f"  Dict keys: {list(entry.keys())}")
        cam0 = entry.get('0')
        if cam0 is not None:
            print(f"\n  --- Inside camera '0' ---")
            print(f"    Keys: {list(cam0.keys())}")
            for k in cam0:
                val = cam0[k]
                print(f"    {k}: type={type(val)}, shape={getattr(val, 'shape', 'N/A')}")
                if hasattr(val, 'shape') and hasattr(val, 'flatten'):
                    print(f"      Sample: {val.flatten()[:10]}")
        else:
            print("    Camera '0' not found in entry.")
    else:
        print(f"  Value: {entry}")




if __name__ == "__main__":
    #debug_npz_structure("data_train_3dhp.npz")
    debug_one_entry("data_train_3dhp.npz")


