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
    # Pick the first key if not specified
    key = entry_key if entry_key is not None else keys[0]
    print(f"\nInspecting key: {key}")
    entry = data[key]
    print(f"  Type: {type(entry)}")
    if isinstance(entry, list):
        print(f"  List length: {len(entry)}")
        if len(entry) > 0:
            print(f"  First element type: {type(entry[0])}")
            if isinstance(entry[0], dict):
                print(f"  Dict keys: {list(entry[0].keys())}")
                for k in entry[0]:
                    val = entry[0][k]
                    print(f"    {k}: type={type(val)}, shape={getattr(val, 'shape', 'N/A')}")
                    # Print a small sample if it's an array
                    if hasattr(val, 'shape') and hasattr(val, 'flatten'):
                        print(f"      Sample: {val.flatten()[:10]}")
    elif isinstance(entry, dict):
        print(f"  Dict keys: {list(entry.keys())}")
        for k in entry:
            val = entry[k]
            print(f"    {k}: type={type(val)}, shape={getattr(val, 'shape', 'N/A')}")
            if hasattr(val, 'shape') and hasattr(val, 'flatten'):
                print(f"      Sample: {val.flatten()[:10]}")
    else:
        print(f"  Value: {entry}")




if __name__ == "__main__":
    #debug_npz_structure("data_train_3dhp.npz")
    debug_one_entry("data_train_3dhp.npz")


