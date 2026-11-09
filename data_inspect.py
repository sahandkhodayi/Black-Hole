import numpy as np

data = np.load("data/photon_trajectories_v2.npz", allow_pickle=False)

print(data.files)
print("number of rays:", len(data["r0"]))
print("number of packed phi samples:", len(data["phi"]))
print("offsets:", data["offsets"][:5])

ray_index = 0
start = data["offsets"][ray_index]
end = data["offsets"][ray_index + 1]

phi = data["phi"][start:end]
u = data["u"][start:end]
v = data["v"][start:end]

print("r0:", data["r0"][ray_index])
print("first state:", u[0], v[0])
print("last state:", u[-1], v[-1])
print("last angle:", phi[-1])
print("stored escape angle:", data["escape_angle"][ray_index])