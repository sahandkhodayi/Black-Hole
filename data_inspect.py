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
radius = np.where(u > 0, 1.0 / u, np.inf)
print(f"phi : {phi[0:30]}")

print(f"first radius: {radius[0]}")
print(f"minimum radius: {radius.min()}")
print(f"maximum radius: {radius.max()}")
print("last angle:", phi[-1])
print("stored escape angle:", data["escape_angle"][ray_index])