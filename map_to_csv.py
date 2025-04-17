

import numpy as np
from skimage.morphology import skeletonize, binary_closing, disk
import matplotlib.pyplot as plt
import yaml
import scipy
from scipy.ndimage import distance_transform_edt as edt
from PIL import Image
import os
import pandas as pd

MAP_NAME = "comp1"
TRACK_WIDTH_MARGIN = 0.0

# Load map image
if os.path.exists(f"maps/{MAP_NAME}.png"):
    map_img_path = f"maps/{MAP_NAME}.png"
elif os.path.exists(f"maps/{MAP_NAME}.pgm"):
    map_img_path = f"maps/{MAP_NAME}.pgm"
else:
    raise Exception("Map not found!")

map_yaml_path = f"maps/{MAP_NAME}.yaml"
raw_map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
raw_map_img = raw_map_img.astype(np.float64)

# Convert grayscale to binary
map_img = raw_map_img.copy()
map_img[map_img <= 210.] = 0
map_img[map_img > 210.] = 1

# Distance transform
dist_transform = scipy.ndimage.distance_transform_edt(map_img)

# Threshold + Morphological cleaning
THRESHOLD = 0.17
centers = dist_transform > THRESHOLD * dist_transform.max()
centers_clean = binary_closing(centers, disk(2))

# Skeletonize
centerline = skeletonize(centers_clean)
centerline_dist = np.where(centerline, dist_transform, 0)

# Find starting point
LEFT_START_Y = centerline_dist.shape[0] // 2
NON_EDGE = 0.0
left_start_y = LEFT_START_Y
left_start_x = 0
while (centerline_dist[left_start_y][left_start_x] == NON_EDGE):
    left_start_x += 1
starting_point = (left_start_x, left_start_y)

# DFS to extract centerline
import sys
sys.setrecursionlimit(20000)

visited = {}
centerline_points = []
track_widths = []
DIRECTIONS = [(0, -1), (-1, 0),  (0, 1), (1, 0), (-1, 1), (-1, -1), (1, 1), (1, -1)]

def dfs(point):
    if point in visited:
        return
    visited[point] = True
    centerline_points.append(np.array(point))
    width_in_pixels = centerline_dist[point[1]][point[0]]
    track_widths.append(np.array([
        width_in_pixels,
        width_in_pixels
    ]))
    for direction in DIRECTIONS:
        nx = point[0] + direction[0]
        ny = point[1] + direction[1]
        if (0 <= nx < centerline_dist.shape[1] and 0 <= ny < centerline_dist.shape[0]):
            if (centerline_dist[ny][nx] != NON_EDGE and (nx, ny) not in visited):
                dfs((nx, ny))

dfs(starting_point)

# Filter out long jumps
filtered_points = [centerline_points[0]]
filtered_widths = [track_widths[0]]
for i in range(1, len(centerline_points)):
    prev = filtered_points[-1]
    curr = centerline_points[i]
    if np.linalg.norm(curr - prev) < 5:  # threshold in pixels
        filtered_points.append(curr)
        filtered_widths.append(track_widths[i])
    else:
        print(f"Skipping jump from {prev} to {curr}")

# Visualize extraction
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
for ax in [ax1, ax2, ax3, ax4]: ax.axis('off')

centerline_img = np.zeros(map_img.shape)
for x, y in filtered_points[:len(filtered_points) // 10]:
    centerline_img[y][x] = 255
ax1.imshow(centerline_img, cmap='Greys', vmax=1, origin='lower')
ax1.set_title("First 10%")

centerline_img = np.zeros(map_img.shape)
for x, y in filtered_points[:len(filtered_points) // 4]:
    centerline_img[y][x] = 255
ax2.imshow(centerline_img, cmap='Greys', vmax=1, origin='lower')
ax2.set_title("First 25%")

centerline_img = np.zeros(map_img.shape)
for x, y in filtered_points[:int(len(filtered_points) / 1.4)]:
    centerline_img[y][x] = 255
ax3.imshow(centerline_img, cmap='Greys', vmax=1, origin='lower')
ax3.set_title("First 50%")

centerline_img = np.zeros(map_img.shape)
for x, y in filtered_points:
    centerline_img[y][x] = 1000
ax4.imshow(centerline_img, cmap='Greys', vmax=1, origin='lower')
ax4.set_title("All points")
fig.tight_layout()

# Prepare for saving
track_widths_np = np.array(filtered_widths)
waypoints = np.array(filtered_points)
data = np.concatenate((waypoints, track_widths_np), axis=1)

# Load map metadata
with open(map_yaml_path, 'r') as yaml_stream:
    map_metadata = yaml.safe_load(yaml_stream)
    map_resolution = map_metadata['resolution']
    origin = map_metadata['origin']

orig_x = origin[0]
orig_y = origin[1]

# Convert all components to meters (including widths!)
transformed_data = data.copy()
transformed_data[:, 0] = transformed_data[:, 0] * map_resolution + orig_x          # x
transformed_data[:, 1] = transformed_data[:, 1] * map_resolution + orig_y          # y
transformed_data[:, 2] = transformed_data[:, 2] * map_resolution - TRACK_WIDTH_MARGIN  # w_tr_right
transformed_data[:, 3] = transformed_data[:, 3] * map_resolution - TRACK_WIDTH_MARGIN  # w_tr_left

# Save CSV
os.makedirs("inputs/tracks", exist_ok=True)
np.savetxt(f"inputs/tracks/{MAP_NAME}.csv", transformed_data, fmt='%0.4f', delimiter=',',
           header='x_m,y_m,w_tr_right_m,w_tr_left_m', comments='')

# OPTIONAL: Visualize the final result
raw_data = pd.read_csv(f"inputs/tracks/{MAP_NAME}.csv")
x = raw_data["x_m"].values
y = raw_data["y_m"].values

x -= orig_x
y -= orig_y
x /= map_resolution
y /= map_resolution

plt.figure(figsize=(10, 10))
plt.imshow(map_img, cmap="gray", origin="lower")
plt.plot(x, y)
plt.title("Final Centerline")
plt.show()

