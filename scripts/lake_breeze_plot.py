"""
This snippet will automatically download the latest NEXRAD KLOT radar image and plot the lake breeze on it. It will then go ahead and tell where to point your instrument with a "T".

"""

import pyart
import boto3
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import torch
import torchvision

from botocore import UNSIGNED
from torchvision.io import decode_image
from torchvision import transforms
from botocore.config import Config
from datetime import datetime, timedelta
from scipy.ndimage import center_of_mass, label

lat_range = [41.1280, 42.5680]
lon_range = [-88.7176, -87.2873]

out_path = '/Users/rjackson/lake_breeze_analysis/'

mobotix_loc = [41.70101404798476, -87.99577278662817]

if __name__ == "__main__":
    right_now = datetime.utcnow()
    yesterday = right_now - timedelta(days=1)
    year = right_now.year
    month = right_now.month
    day = right_now.day
    
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = 'noaa-nexrad-level2'
    radar = "KLOT"
    prefix = f'{year}/{month:02d}/{day:02d}/{radar}'
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    file_list = [x['Key'] for x in response['Contents']]
    
    # Find yesterday's scans
    prefix = f'{yesterday.year}/{yesterday.month:02d}/{yesterday.day:02d}/{radar}'
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    file_list = file_list + [x['Key'] for x in response['Contents']]
    
    time_list = []
    for filepath in file_list:
        name = filepath.split("/")[-1]
        if name[-3:] == "MDM":
            time_list.append(datetime.strptime(name, f"{radar}%Y%m%d_%H%M%S_V06_MDM"))
        else:
            time_list.append(datetime.strptime(name, f"{radar}%Y%m%d_%H%M%S_V06"))
        
    time_list = np.array(time_list)
    cur_time = time_list[np.argmin(np.abs(time_list - right_now))]
    path = "s3://noaa-nexrad-level2/" + file_list[np.argmin(np.abs(time_list - right_now))]
    print(path)
    cur_radar = pyart.io.read_nexrad_archive(path)
    dpi = 150
    disp = pyart.graph.RadarMapDisplay(cur_radar)
    fig, ax = plt.subplots(1, 1, figsize=(256/dpi, 256/dpi),
            subplot_kw=dict(projection=ccrs.PlateCarree(), frameon=False))
    
    disp.plot_ppi_map('reflectivity', sweep=1, min_lon=lon_range[0],
            ax=ax, max_lon=lon_range[1], min_lat=lat_range[0], max_lat=lat_range[1],
            embellish=False, vmin=-30, vmax=60, cmap='Spectral_r',
            add_grid_lines=False, colorbar_flag=False, title_flag=False)
    ax.set_axis_off()
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    fig.savefig('temp.png', dpi=150)

    # Transform image
    image = decode_image('temp.png')
    image = image[:3, :, :].float()
    transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image = torch.stack([transform(image)])
    model = torch.load('epoch99.pickle', weights_only=False, map_location=torch.device('cpu'))
    os.remove('temp.png')
    
    mask = model(image)['out'].detach().numpy()
    mask = mask[0].argmax(axis=0)

    # Filter out small regions that are not lake breezes
    labels, num_features = label(mask)
    area_threshold = 20
    print(num_features)
    for i in range(num_features):
        area = mask[labels == i+1].sum()
        if area < area_threshold:
            mask[labels == i+1] = 0

    # Decide where to point the instrument
    lats = np.linspace(lat_range[1], lat_range[0], mask.shape[1])
    lons = np.linspace(lon_range[1], lon_range[0], mask.shape[0])
    lat_index = np.argmin(np.abs(lats - mobotix_loc[1]))
    lon_index = np.argmin(np.abs(lons - mobotix_loc[0]))
    center = center_of_mass(mask)
    angle = np.atan2(-(center[1] - lat_index), (center[0] - lon_index))
    if angle < 0:
        angle = angle + 2 * np.pi
    deg_angle = np.rad2deg(angle)
    if np.isnan(deg_angle):
        print("We are not triggering the mobotix")
    else:
        print(f"We will point the mobotix at {deg_angle:.2f} degrees")
   
    lons, lats = np.meshgrid(lons, lats, indexing='ij')
    lat_index = int(center[1])
    lon_index = int(center[0])
    disp = pyart.graph.RadarMapDisplay(cur_radar)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5),
            subplot_kw=dict(projection=ccrs.PlateCarree()))
    disp.plot_ppi_map('reflectivity', sweep=1, min_lon=lon_range[0],
            ax=ax, max_lon=lon_range[1], min_lat=lat_range[0], max_lat=lat_range[1],
            lat_lines=[41.2, 41.4, 41.6, 41.8, 42, 42.2, 42.4],
            lon_lines=[-88.4, -88.2, -88, -87.8, -87.6],
            vmin=-30, vmax=60, cmap='Spectral_r')
            
    ax.coastlines()
    ax.add_feature(cfeature.STATES)
    ax.contour(lons, lats, mask, levels=1)
    ax.text(lons[lat_index, lon_index], lats[lat_index, lon_index], 'T')
    ax.text(mobotix_loc[1], mobotix_loc[0], 'X')
    date_str = cur_time.strftime("%Y%m%d.%H%M%S")
    fig.savefig(os.path.join(out_path, f'KLOT_lake_breeze{date_str}.png'))
    

    