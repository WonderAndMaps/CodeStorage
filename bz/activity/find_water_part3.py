INSTANCE_ID = 'de419661-122d-42cc-8366-8b1fd29f4913'

import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, CustomUrlParam, get_area_dates
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest

        
def plot_previews(data, dates, cols = 4, figsize=(15,15), denom=1.):
    """
    Utility to plot small "true color" previews.
    """
    width = data[-1].shape[1]
    height = data[-1].shape[0]
    
    rows = data.shape[0] // cols + (1 if data.shape[0] % cols else 0)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    for index, ax in enumerate(axs.flatten()):
        if index < data.shape[0]:
            caption = str(index)+': '+dates[index].strftime('%Y-%m-%d')
            #ax.set_axis_off()
            ax.imshow(data[index] / denom, vmin=0.0, vmax=1.0)
            ax.text(0, -2, caption, fontsize=12, color='g')
        else:
            ax.set_axis_off()


def get_data(lefttoplon,lefttoplat,rightbtmlon,rightbtmlat,time,maxcc=1,NDWI=True):
    """
    Return the NDWI imageries given a bounding box and time
    """
    if abs(lefttoplon) > 180 or abs(rightbtmlon) > 180:
        print("wrong longitude")
        return None
    if abs(lefttoplat) > 90 or abs(rightbtmlat) > 90:
        print("wrong latitude")
        return None
    
    if NDWI:
        layer='NDWI'
    else:
        layer='TRUE_COLOR'
    
    desired_coords_wgs84 = [lefttoplon,lefttoplat,rightbtmlon,rightbtmlat]        
    desired_bbox = BBox(bbox=desired_coords_wgs84, crs=CRS.WGS84)
    
    wms_request = WmsRequest(layer=layer,
                         bbox=desired_bbox,
                         time=time,
                         maxcc=maxcc,
                         width=100, height=100,
                         instance_id=INSTANCE_ID)
    
    wms_img = wms_request.get_data()
    return wms_img,wms_request.get_dates()


def get_cloud(lefttoplon,lefttoplat,rightbtmlon,rightbtmlat,time):
    """
    Return cloud masks given a bounding box and time
    """
    if abs(lefttoplon) > 180 or abs(rightbtmlon) > 180:
        print("wrong longitude")
        return None
    if abs(lefttoplat) > 90 or abs(rightbtmlat) > 90:
        print("wrong latitude")
        return None
    
    bands_script = 'return [B01,B02,B04,B05,B08,B8A,B09,B10,B11,B12]'
    desired_coords_wgs84 = [lefttoplon,lefttoplat,rightbtmlon,rightbtmlat]        
    desired_bbox = BBox(bbox=desired_coords_wgs84, crs=CRS.WGS84)
    
    wms_bands_request = WmsRequest(layer='TRUE_COLOR',
                                   custom_url_params={CustomUrlParam.EVALSCRIPT: bands_script},
                                   bbox=desired_bbox, 
                                   time=time, 
                                   width=100, height=100,
                                   image_format=MimeType.TIFF_d32f,
                                   instance_id=INSTANCE_ID)


    all_cloud_masks = CloudMaskRequest(ogc_request=wms_bands_request, threshold=0.4)
    cloud_dates = all_cloud_masks.get_dates()
    cloud_masks = all_cloud_masks.get_cloud_masks(threshold=0.4)
    
    return cloud_masks, cloud_dates



def water_and_cloud_preview(lefttoplon,lefttoplat,rightbtmlon,rightbtmlat,time):
    imgs_ndwi,imgs_ndwi_dates = get_data(lefttoplon,lefttoplat,rightbtmlon,rightbtmlat,time)
    imgs_ndwi = np.asarray(imgs_ndwi)

    cloud_masks,cloud_dates = get_cloud(lefttoplon,lefttoplat,rightbtmlon,rightbtmlat,time)
    
    imgs_true,imgs_true_dates = get_data(lefttoplon,lefttoplat,rightbtmlon,rightbtmlat,time,NDWI=False)
    imgs_true = np.asarray(imgs_true)

    plot_previews(imgs_ndwi,imgs_ndwi_dates,figsize=(15,10))
    plot_previews(cloud_masks,cloud_dates,figsize=(15,10))
    plot_previews(imgs_true,imgs_true_dates,figsize=(15,10),denom=255.)

def find_arrival(status):
    """
    Find water's first arrival. The first digit indicates cloud 
    status. The second digit indicates water status.
    In short, find the first 01 after 00.
    """
    obs = np.array(status).astype(np.int32)
    no_cloud = np.where(obs<10)
    no_cloud_obs = obs[no_cloud]
    
    if no_cloud_obs[0]==1:
        print("It is there since the beginning.")
        return -1
    if len(no_cloud_obs)==1:
        print("There is only one no-cloud observation.")
        return -1

    prev_obs = np.concatenate((np.zeros(1),obs[:-1])).astype(np.int32)
    result = np.array(np.where(( (obs-prev_obs)==1) * (obs<10) ))[0].tolist()
    
    if len(result)==0:
        print("Can't find arrival.")
    return result

def lonlat_to_xy(target_lon,target_lat, lefttoplon,lefttoplat,rightbtmlon,rightbtmlat, fac=100.):
    map_y = (target_lat-lefttoplat)/(rightbtmlat-lefttoplat)*fac
    map_x = (target_lon-lefttoplon)/(rightbtmlon-lefttoplon)*fac
    return map_x,map_y
#=============================================================

lefttoplon,lefttoplat,rightbtmlon,rightbtmlat,time = -102.02986,31.94470,-101.98102,31.92329,('2018-01-01','2018-08-31')

target_lon,target_lat = -101.98795528,31.934979860000002
mapx,mapy = lonlat_to_xy(target_lon,target_lat,lefttoplon,lefttoplat,rightbtmlon,rightbtmlat)


# Preview
#water_and_cloud_preview(lefttoplon,lefttoplat,rightbtmlon,rightbtmlat,time)


# Find water arrival date
imgs_ndwi,imgs_ndwi_dates = get_data(lefttoplon,lefttoplat,rightbtmlon,rightbtmlat,time)
imgs_ndwi = np.asarray(imgs_ndwi)
cloud_masks,cloud_dates = get_cloud(lefttoplon,lefttoplat,rightbtmlon,rightbtmlat,time)
wc = cloud_masks*10+(imgs_ndwi!=0)

arrival_ind = find_arrival(wc[:,43,27])
for i in arrival_ind:
    print(cloud_dates[i])


plt.imshow(imgs_ndwi[-1,:,:])
plt.scatter(274,mapy)

imgs_ndwi[5,427,274]

find_arrival(wc[:,274,int(mapy)])

find_arrival([0,0,1,10,10,1,1,1,1,0,0,0,10,11,10,11,10,0,1])
