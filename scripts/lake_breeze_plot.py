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
import sys

from botocore import UNSIGNED
from torchvision.io import decode_image
from torchvision import transforms
from torch.nn import Identity
from torch.nn.modules.conv import Conv2d
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead, ASPP, ASPPConv, ASPPPooling
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
from torch.nn.modules.container import Sequential, ModuleList
from torchvision.models.resnet import Bottleneck
from torchvision.models._utils import IntermediateLayerGetter
from botocore.config import Config
from datetime import datetime, timedelta
from scipy.ndimage import center_of_mass, label

lat_range = [41.1280, 42.5680]
lon_range = [-88.7176, -87.2873]
if len(sys.argv) > 1:
    inp_time = datetime.strptime(sys.argv[1], '%Y%m%d.%H%M%S')
else:
    inp_time = None

out_path = '/nfs/gce/projects/crocus/lake_breeze_ids'

mobotix_loc = [41.70101404798476, -87.99577278662817]

if __name__ == "__main__":
    if inp_time is None:
        right_now = datetime.utcnow()
    else:
        right_now = inp_time
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
    out_path = os.path.join(out_path, f'{year}/{year}{month:02d}{day}/')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    time_list = []
    for filepath in file_list:
        name = filepath.split("/")[-1]
        if name[-3:] == "MDM":
            time_list.append(
                datetime.strptime(name, f"{radar}%Y%m%d_%H%M%S_V06_MDM"))
        else:
            time_list.append(
                datetime.strptime(name, f"{radar}%Y%m%d_%H%M%S_V06"))
        
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

    # Model #1 - Bobby's initial UNet
    model = torch.load('epoch99.pickle', weights_only=False,
        map_location=torch.device('cpu'))
    os.remove('temp.png')
    
    mask = model(image)['out'].detach().numpy()
    mask = mask[0].argmax(axis=0)
    mask = mask

    # Filter out small regions that are not lake breezes
    labels, num_features = label(mask)
    area_threshold = 20
    print(num_features)
    largest_area = -99999
    largest_index = 0
    for i in range(num_features):
        area = mask[labels == i+1].sum()
        if area > largest_area:
            area = largest_area
            largest_index = i+1
        if area < area_threshold:
            mask[labels == i+1] = 0
    mask = mask.T
    mask = mask[:, ::-1]
    
    # Decide where to point the instrument
    lats = np.linspace(lat_range[0], lat_range[1], mask.shape[1])
    lons = np.linspace(lon_range[0], lon_range[1], mask.shape[0])
    lat_index = np.argmin(np.abs(lats - mobotix_loc[1]))
    lon_index = np.argmin(np.abs(lons - mobotix_loc[0]))

    # Point to the center of the largest lake breeze region
    mask_largest = np.where(mask == largest_index, mask, 0)
    center = center_of_mass(mask_largest)
    angle = np.atan2(-(center[1] - lat_index), (center[0] - lon_index))
    if angle < 0:
        angle = angle + 2 * np.pi
    deg_angle = np.rad2deg(angle)
    
    # Decide where to point the instrument
    lats = np.linspace(lat_range[0], lat_range[1], mask.shape[1])
    lons = np.linspace(lon_range[0], lon_range[1], mask.shape[0])
    lat_index = np.argmin(np.abs(lats - mobotix_loc[1]))
    lon_index = np.argmin(np.abs(lons - mobotix_loc[0]))
    
    if np.isnan(deg_angle):
        print("We are not triggering the mobotix")
    else:
        print(f"We will point the mobotix at {deg_angle:.2f} degrees")
    

    lons, lats = np.meshgrid(lons, lats, indexing='xy')
    if not np.isnan(center[1]):
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
    if not np.isnan(center[1]):
        ax.text(lons[lat_index, lon_index], lats[lat_index, lon_index], 'T')
    ax.text(mobotix_loc[1], mobotix_loc[0], 'X')
    date_str = cur_time.strftime("%Y%m%d.%H%M%S")
    out_path_bobby_unet = os.path.join(out_path, 'bobby_unet')
    if not os.path.exists(out_path_bobby_unet):
        os.makedirs(out_path_bobby_unet)
    fig.savefig(os.path.join(out_path_bobby_unet, f'KLOT_lake_breeze{date_str}.png'))
    plt.close(fig)

    # Load Seongha's models
    model_deeplabv3_resnet101 = torchvision.models.segmentation.deeplabv3_resnet101(
        weights='DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1')
    model_deeplabv3_resnet50 = torchvision.models.segmentation.deeplabv3_resnet50(
        weights='DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1')
    model_fcn_resnet101 = torchvision.models.segmentation.fcn_resnet101(
        weights='FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1')
    model_fcn_resnet50 = torchvision.models.segmentation.fcn_resnet50(
        weights='FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1') 

    in_channels = 2048
    inter_channels = 512
    channels = 2
    fcn_new_last_layer = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        torch.nn.BatchNorm2d(inter_channels),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Conv2d(inter_channels, channels, 1),
    )
    model_fcn_resnet101.classifier = fcn_new_last_layer
    model_fcn_resnet101.aux_classifier = Identity()
    model_fcn_resnet50.classifier = fcn_new_last_layer
    model_fcn_resnet50.aux_classifier = Identity()

    d_in_channels = 1024
    d_inter_channels = 256
    deeplab_new_last_layer = torch.nn.Sequential(
        torch.nn.Conv2d(d_in_channels, d_inter_channels, 3, padding=1, bias=False),
        torch.nn.BatchNorm2d(d_inter_channels),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Conv2d(d_inter_channels, channels, 1),
    )

    model_deeplabv3_resnet101.aux_classifier = deeplab_new_last_layer
    model_deeplabv3_resnet50.aux_classifier = deeplab_new_last_layer

    model_unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                    in_channels=3, out_channels=1,
                    init_features=32, pretrained=True)
    new_last_layer = torch.nn.Sequential(
        torch.nn.Conv2d(32, 2, 1),
    )

    model_unet.conv = new_last_layer
    model_unet.load_state_dict(torch.load('../models/lakebreeze_best_model_unet.pth',
        map_location=torch.device('cpu'), weights_only=True))

    # Load our model state dictionary
    with torch.serialization.safe_globals(
        [DeepLabV3, BatchNorm2d, Conv2d, IntermediateLayerGetter,
         ReLU, MaxPool2d, Sequential, Bottleneck, DeepLabHead, ASPP,
         ModuleList, ASPPConv, ASPPPooling, AdaptiveAvgPool2d, Dropout]) as globby:
        model_fcn_resnet50= torch.load('../models/lakebreeze_best_model_fcn_resnet50.pth',
            map_location=torch.device('cpu'), weights_only=True)
        model_fcn_resnet101 = torch.load('../models/lakebreeze_best_model_fcn_resnet101.pth',
            map_location=torch.device('cpu'), weights_only=True)
        model_deeplabv3_resnet101 = torch.load('../models/lakebreeze_best_model_deeplabv3_resnet101.pth',
            map_location=torch.device('cpu'), weights_only=True)
        model_deeplabv3_resnet50 = torch.load('../models/lakebreeze_best_model_deeplabv3_resnet50.pth',
            map_location=torch.device('cpu'), weights_only=True)

    # Plot UNet model prediction and Mobotix pointing direction
    mask = model_unet(image).detach().numpy()
    mask = mask[0].argmax(axis=0)
    mask = mask

    # Filter out small regions that are not lake breezes
    labels, num_features = label(mask)
    area_threshold = 20
    print(num_features)
    largest_area = -99999
    largest_index = 0
    for i in range(num_features):
        area = mask[labels == i+1].sum()
        if area > largest_area:
            area = largest_area
            largest_index = i+1
        if area < area_threshold:
            mask[labels == i+1] = 0
    mask = mask.T
    mask = mask[:, ::-1]

    # Point to the center of the largest lake breeze region
    mask_largest = np.where(mask == largest_index, mask, 0)
    center = center_of_mass(mask_largest)
    angle = np.atan2(-(center[1] - lat_index), (center[0] - lon_index))
    if angle < 0:
        angle = angle + 2 * np.pi
    deg_angle = np.rad2deg(angle)

     # Decide where to point the instrument
    lats = np.linspace(lat_range[0], lat_range[1], mask.shape[1])
    lons = np.linspace(lon_range[0], lon_range[1], mask.shape[0])
    lat_index = np.argmin(np.abs(lats - mobotix_loc[1]))
    lon_index = np.argmin(np.abs(lons - mobotix_loc[0]))
    
    if np.isnan(deg_angle):
        print("We are not triggering the mobotix")
    else:
        print(f"We will point the mobotix at {deg_angle:.2f} degrees")
    
    lons, lats = np.meshgrid(lons, lats, indexing='xy')
    if not np.isnan(center[1]):
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
    if not np.isnan(center[1]):
        ax.text(lons[lat_index, lon_index], lats[lat_index, lon_index], 'T')
    ax.text(mobotix_loc[1], mobotix_loc[0], 'X')
    date_str = cur_time.strftime("%Y%m%d.%H%M%S")
    out_path_bobby_unet = os.path.join(out_path, 'model_unet')
    if not os.path.exists(out_path_bobby_unet):
        os.makedirs(out_path_bobby_unet)
    fig.savefig(os.path.join(out_path_bobby_unet, f'KLOT_lake_breeze{date_str}.png'))
    plt.close(fig)

    # Plot FCN-Resnet50 model prediction and Mobotix pointing direction
    mask = model_fcn_resnet50(image)['out'].detach().numpy()
    mask = mask[0].argmax(axis=0)
    mask = mask

    # Filter out small regions that are not lake breezes
    labels, num_features = label(mask)
    area_threshold = 20
    print(num_features)
    largest_area = -99999
    largest_index = 0
    for i in range(num_features):
        area = mask[labels == i+1].sum()
        if area > largest_area:
            area = largest_area
            largest_index = i+1
        if area < area_threshold:
            mask[labels == i+1] = 0
    mask = mask.T
    mask = mask[:, ::-1]

    # Point to the center of the largest lake breeze region
    mask_largest = np.where(mask == largest_index, mask, 0)
    center = center_of_mass(mask_largest)
    angle = np.atan2(-(center[1] - lat_index), (center[0] - lon_index))
    if angle < 0:
        angle = angle + 2 * np.pi
    deg_angle = np.rad2deg(angle)

     # Decide where to point the instrument
    lats = np.linspace(lat_range[0], lat_range[1], mask.shape[1])
    lons = np.linspace(lon_range[0], lon_range[1], mask.shape[0])
    lat_index = np.argmin(np.abs(lats - mobotix_loc[1]))
    lon_index = np.argmin(np.abs(lons - mobotix_loc[0]))
    
    if np.isnan(deg_angle):
        print("We are not triggering the mobotix")
    else:
        print(f"We will point the mobotix at {deg_angle:.2f} degrees")
    
    lons, lats = np.meshgrid(lons, lats, indexing='xy')
    if not np.isnan(center[1]):
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
    if not np.isnan(center[1]):
        ax.text(lons[lat_index, lon_index], lats[lat_index, lon_index], 'T')
    ax.text(mobotix_loc[1], mobotix_loc[0], 'X')
    date_str = cur_time.strftime("%Y%m%d.%H%M%S")
    out_path_bobby_unet = os.path.join(out_path, 'model_fcn_resnet50')
    if not os.path.exists(out_path_bobby_unet):
        os.makedirs(out_path_bobby_unet)
    fig.savefig(os.path.join(out_path_bobby_unet, f'KLOT_lake_breeze{date_str}.png'))
    plt.close(fig)
    
    # Plot FCN-Resnet50 model prediction and Mobotix pointing direction
    mask = model_fcn_resnet101(image)['out'].detach().numpy()
    mask = mask[0].argmax(axis=0)
    mask = mask

    # Filter out small regions that are not lake breezes
    labels, num_features = label(mask)
    area_threshold = 20
    print(num_features)
    largest_area = -99999
    largest_index = 0
    for i in range(num_features):
        area = mask[labels == i+1].sum()
        if area > largest_area:
            area = largest_area
            largest_index = i+1
        if area < area_threshold:
            mask[labels == i+1] = 0
    mask = mask.T
    mask = mask[:, ::-1]

    # Point to the center of the largest lake breeze region
    mask_largest = np.where(mask == largest_index, mask, 0)
    center = center_of_mass(mask_largest)
    angle = np.atan2(-(center[1] - lat_index), (center[0] - lon_index))
    if angle < 0:
        angle = angle + 2 * np.pi
    deg_angle = np.rad2deg(angle)

     # Decide where to point the instrument
    lats = np.linspace(lat_range[0], lat_range[1], mask.shape[1])
    lons = np.linspace(lon_range[0], lon_range[1], mask.shape[0])
    lat_index = np.argmin(np.abs(lats - mobotix_loc[1]))
    lon_index = np.argmin(np.abs(lons - mobotix_loc[0]))
    
    if np.isnan(deg_angle):
        print("We are not triggering the mobotix")
    else:
        print(f"We will point the mobotix at {deg_angle:.2f} degrees")
    
    lons, lats = np.meshgrid(lons, lats, indexing='xy')
    if not np.isnan(center[1]):
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
    if not np.isnan(center[1]):
        ax.text(lons[lat_index, lon_index], lats[lat_index, lon_index], 'T')
    ax.text(mobotix_loc[1], mobotix_loc[0], 'X')
    date_str = cur_time.strftime("%Y%m%d.%H%M%S")
    out_path_bobby_unet = os.path.join(out_path, 'model_fcn_resnet101')
    if not os.path.exists(out_path_bobby_unet):
        os.makedirs(out_path_bobby_unet)
    fig.savefig(os.path.join(out_path_bobby_unet, f'KLOT_lake_breeze{date_str}.png'))
    plt.close(fig)

    # Plot Deeplab-Resnet50 model prediction and Mobotix pointing direction
    mask = model_deeplabv3_resnet50(image)['out'].detach().numpy()
    mask = mask[0].argmax(axis=0)
    mask = mask

    # Filter out small regions that are not lake breezes
    labels, num_features = label(mask)
    area_threshold = 20
    print(num_features)
    largest_area = -99999
    largest_index = 0
    for i in range(num_features):
        area = mask[labels == i+1].sum()
        if area > largest_area:
            area = largest_area
            largest_index = i+1
        if area < area_threshold:
            mask[labels == i+1] = 0
    mask = mask.T
    mask = mask[:, ::-1]

    # Point to the center of the largest lake breeze region
    mask_largest = np.where(mask == largest_index, mask, 0)
    center = center_of_mass(mask_largest)
    angle = np.atan2(-(center[1] - lat_index), (center[0] - lon_index))
    if angle < 0:
        angle = angle + 2 * np.pi
    deg_angle = np.rad2deg(angle)

     # Decide where to point the instrument
    lats = np.linspace(lat_range[0], lat_range[1], mask.shape[1])
    lons = np.linspace(lon_range[0], lon_range[1], mask.shape[0])
    lat_index = np.argmin(np.abs(lats - mobotix_loc[1]))
    lon_index = np.argmin(np.abs(lons - mobotix_loc[0]))
    
    if np.isnan(deg_angle):
        print("We are not triggering the mobotix")
    else:
        print(f"We will point the mobotix at {deg_angle:.2f} degrees")
    
    lons, lats = np.meshgrid(lons, lats, indexing='xy')
    if not np.isnan(center[1]):
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
    if not np.isnan(center[1]):
        ax.text(lons[lat_index, lon_index], lats[lat_index, lon_index], 'T')
    ax.text(mobotix_loc[1], mobotix_loc[0], 'X')
    date_str = cur_time.strftime("%Y%m%d.%H%M%S")
    out_path_bobby_unet = os.path.join(out_path, 'model_deeplabv3_resnet50')
    if not os.path.exists(out_path_bobby_unet):
        os.makedirs(out_path_bobby_unet)
    fig.savefig(os.path.join(out_path_bobby_unet, f'KLOT_lake_breeze{date_str}.png'))
    plt.close(fig)

    # Plot Deeplab-Resnet101 model prediction and Mobotix pointing direction
    mask = model_deeplabv3_resnet101(image)['out'].detach().numpy()
    mask = mask[0].argmax(axis=0)
    mask = mask

    # Filter out small regions that are not lake breezes
    labels, num_features = label(mask)
    area_threshold = 20
    print(num_features)
    largest_area = -99999
    largest_index = 0
    for i in range(num_features):
        area = mask[labels == i+1].sum()
        if area > largest_area:
            area = largest_area
            largest_index = i+1
        if area < area_threshold:
            mask[labels == i+1] = 0
    mask = mask.T
    mask = mask[:, ::-1]

    # Point to the center of the largest lake breeze region
    mask_largest = np.where(mask == largest_index, mask, 0)
    center = center_of_mass(mask_largest)
    angle = np.atan2(-(center[1] - lat_index), (center[0] - lon_index))
    if angle < 0:
        angle = angle + 2 * np.pi
    deg_angle = np.rad2deg(angle)

     # Decide where to point the instrument
    lats = np.linspace(lat_range[0], lat_range[1], mask.shape[1])
    lons = np.linspace(lon_range[0], lon_range[1], mask.shape[0])
    lat_index = np.argmin(np.abs(lats - mobotix_loc[1]))
    lon_index = np.argmin(np.abs(lons - mobotix_loc[0]))
    
    if np.isnan(deg_angle):
        print("We are not triggering the mobotix")
    else:
        print(f"We will point the mobotix at {deg_angle:.2f} degrees")
    
    lons, lats = np.meshgrid(lons, lats, indexing='xy')
    if not np.isnan(center[1]):
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
    if not np.isnan(center[1]):
        ax.text(lons[lat_index, lon_index], lats[lat_index, lon_index], 'T')
    ax.text(mobotix_loc[1], mobotix_loc[0], 'X')
    date_str = cur_time.strftime("%Y%m%d.%H%M%S")
    out_path_bobby_unet = os.path.join(out_path, 'model_deeplabv3_resnet101')
    if not os.path.exists(out_path_bobby_unet):
        os.makedirs(out_path_bobby_unet)
    fig.savefig(os.path.join(out_path_bobby_unet, f'KLOT_lake_breeze{date_str}.png'))
    plt.close(fig)





    

    
