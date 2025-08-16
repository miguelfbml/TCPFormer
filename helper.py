import pickle

with open("data/motion3d/H36M-81/test/00000015.pkl", "rb") as file:
    loaded_data = pickle.load(file)
print(loaded_data)