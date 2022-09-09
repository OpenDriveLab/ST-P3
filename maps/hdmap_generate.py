import numpy as np
import cv2 as cv
import json
from collections import deque
from pathlib import Path
import h5py
import os
import tqdm

classes = {
  0: [0, 0, 0],         # unlabeled
  1: [0, 0, 0],         # building
  2: [0, 0, 0],         # fence
  3: [0, 0, 0],         # other
  4: [0, 255, 0],       # pedestrian
  5: [0, 0, 0],         # pole
  6: [157, 234, 50],    # road line
  7: [128, 64, 128],    # road
  8: [255, 255, 255],   # sidewalk
  9: [0, 0, 0],         # vegetation
  10: [0, 0, 255],      # vehicle
  11: [0, 0, 0],        # wall
  12: [0, 0, 0],        # traffic sign
  13: [0, 0, 0],        # sky
  14: [0, 0, 0],        # ground
  15: [0, 0, 0],        # bridge
  16: [0, 0, 0],        # rail track
  17: [0, 0, 0],        # guard rail
  18: [0, 0, 0],        # traffic light
  19: [0, 0, 0],        # static
  20: [0, 0, 0],        # dynamic
  21: [0, 0, 0],        # water
  22: [0, 0, 0],        # terrain
  23: [255, 0, 0],      # red light
  24: [0, 0, 0],        # yellow light #TODO should be red
  25: [0, 0, 0],        # green light
  26: [157, 234, 50],   # stop sign
  27: [157, 234, 50],   # stop line marking
    
}


COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)

pixels_per_meter = 5
width = 512
pixels_ev_to_bottom = 256

def tint(color, factor):
	r, g, b = color
	r = int(r + (255-r) * factor)
	g = int(g + (255-g) * factor)
	b = int(b + (255-b) * factor)
	r = min(r, 255)
	g = min(g, 255)
	b = min(b, 255)
	return (r, g, b)

def get_warp_transform(ev_loc, ev_rot, world_offset):
		ev_loc_in_px = world_to_pixel(ev_loc, world_offset)
		yaw = np.deg2rad(ev_rot)

		forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
		right_vec = np.array([np.cos(yaw + 0.5*np.pi), np.sin(yaw + 0.5*np.pi)])

		bottom_left = ev_loc_in_px - pixels_ev_to_bottom * forward_vec - (0.5*width) * right_vec
		top_left = ev_loc_in_px + (width-pixels_ev_to_bottom) * forward_vec - (0.5*width) * right_vec
		top_right = ev_loc_in_px + (width-pixels_ev_to_bottom) * forward_vec + (0.5*width) * right_vec

		src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
		dst_pts = np.array([[0, width-1],
							[0, 0],
							[width-1, 0]], dtype=np.float32)
		return cv.getAffineTransform(src_pts, dst_pts)

def world_to_pixel(location, world_offset, projective=False):
	"""Converts the world coordinates to pixel coordinates"""
	x = pixels_per_meter * (location[0] - world_offset[0])
	y = pixels_per_meter * (location[1] - world_offset[1])

	if projective:
		p = np.array([x, y, 1], dtype=np.float32)
	else:
		p = np.array([x, y], dtype=np.float32)
	return p

Town2map = {
	"town01": "Town01.h5",
	"town02": "Town02.h5",
	"town03": "Town03.h5",
	"town04": "Town04.h5",
	"town05": "Town05.h5",
	"town06": "Town06.h5",
	"town07": "Town07.h5",
	"town10": "Town10HD.h5",
	
}

root = ""
map_path = ""
town_list = list(os.listdir(root))
# town_list = town_list[18:]
# town_list = ["town02_short","town02_tiny"]
for town in tqdm.tqdm(town_list):

	maps_h5_path = os.path.join(map_path, Town2map[town[:6]])

	with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf: 
		road = np.array(hf['road'], dtype=np.uint8)
		lane_marking_yellow_broken = np.array(hf['lane_marking_yellow_broken'], dtype=np.uint8)
		lane_marking_yellow_solid = np.array(hf['lane_marking_yellow_solid'], dtype=np.uint8)
		lane_marking_white_broken = np.array(hf['lane_marking_white_broken'], dtype=np.uint8)
		lane_marking_white_solid = np.array(hf['lane_marking_white_solid'], dtype=np.uint8)
		world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)


	town_folder = os.path.join(root, town)
	route_list = [route for route in os.listdir(town_folder) if os.path.isdir(os.path.join(town_folder, route)) ]

	for route in tqdm.tqdm(route_list):
		route_folder = os.path.join(town_folder, route)
		os.makedirs(os.path.join(route_folder, "hdmap"), exist_ok=True)
		measurement_folder = os.path.join(route_folder, "meta")
		measurement_files = os.listdir(measurement_folder)
		for measurement in measurement_files:
			with open(os.path.join(measurement_folder, measurement), "r") as read_file:
				measurement_data = json.load(read_file)
				x = measurement_data['x']
				y = measurement_data['y']
				
				theta = measurement_data['theta']
				if np.isnan(theta):
					theta = 0


				ev_loc = [y , -x]
				ev_rot = np.rad2deg(theta) - 90

				M_warp = get_warp_transform(ev_loc, ev_rot, world_offset)
				road_mask = cv.warpAffine(road, M_warp, (width, width)).astype(np.bool)
				lane_mask_white_broken = cv.warpAffine(lane_marking_white_broken, M_warp,
														(width, width)).astype(np.bool)
				lane_mask_white_solid = cv.warpAffine(lane_marking_white_solid, M_warp,
														(width, width)).astype(np.bool)
				lane_mask_yellow_broken = cv.warpAffine(lane_marking_yellow_broken, M_warp,
														(width, width)).astype(np.bool)
				lane_mask_yellow_solid = cv.warpAffine(lane_marking_yellow_solid, M_warp,
														(width, width)).astype(np.bool)

				image = np.zeros([width, width, 3], dtype=np.uint8)
				image[road_mask] = COLOR_ALUMINIUM_5
				image[lane_mask_white_broken] = COLOR_MAGENTA
				image[lane_mask_white_solid] = COLOR_MAGENTA
				image[lane_mask_yellow_broken] = COLOR_MAGENTA
				image[lane_mask_yellow_solid] = COLOR_MAGENTA
				cv.imwrite(os.path.join(route_folder, "hdmap", measurement.replace('json', 'png')), image)









# image[lane_mask_broken] = COLOR_MAGENTA_2

# cv.imwrite("hdmap.png", image)


# seg = cv.imread("0009.png")[:, :, 2]
# result = np.zeros((seg.shape[0], seg.shape[1], 3))
# for key, value in classes.items():
# 	result[np.where(seg == key)] = value

# cv.imwrite("seg_vis.png", result)