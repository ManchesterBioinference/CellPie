import cv2
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from squidpy.im._feature_mixin import FeatureMixin
from sklearn.preprocessing import MinMaxScaler
import spatialdata as sd
import spatialdata_io
import json


# the features_histogram function has been taken from: https://github.com/scverse/squidpy


def features_histogram(
    arr: np.ndarray,
    layer: str = 'image_array',
    library_id: str | None = None,
    spot_scale=None,
    feature_name: str = "histogram",
    channels: list[int] | None = None,
    bins: int = 100,
    v_range: tuple[int, int] | None = None,
) -> dict:
    """
    Compute histogram counts of color channel values.

    Returns one feature per bin and channel.

    Parameters
    ----------
    arr : np.ndarray
        The image data array.
    layer : str, optional
        The layer of the image to process, by default 'image_array'.
    library_id : str | None, optional
        Library ID to select specific image(s), by default None.
    feature_name : str, optional
        The base name for the feature keys, by default "histogram".
    channels : list[int] | None, optional
        List of channels to process, by default None which means all channels.
    bins : int, optional
        Number of binned value intervals, by default 10.
    v_range : tuple[int, int] | None, optional
        Range on which values are binned. If `None`, use the whole image range.

    Returns
    -------
    dict
        Returns features with the following keys for each channel `c` in ``channels``:

        - ``'{feature_name}_ch-{c}_bin-{i}'`` - the histogram counts for each bin `i` in ``bins``.
    """
    # If channels are not provided, use all available channels
    if channels is None:
        channels = range(arr.shape[-1])  # Assuming the last dimension is channels

    # Ensure channels is a non-empty sequence
    if len(channels) == 0:
        raise ValueError("Channels must be a non-empty sequence")

    # If v_range is None, use the whole-image range
    if v_range is None:
        v_range = (np.min(arr), np.max(arr))
    
    features = {}
    for c in channels:
        hist, _ = np.histogram(arr[..., c], bins=bins, range=v_range, weights=None, density=False)
        for i, count in enumerate(hist):
            features[f"{feature_name}_ch-{c}_bin-{i}_{spot_scale}"] = count

    return features

# Function to flatten the histogram features
def flatten_features(features, scale):
    flattened = {}
    for key, value in features.items():
        flattened[f"{key}_scale{scale}"] = value
    return flattened

def extract_features(adata, img_path, spot_scale: list, bins: int = 10, scale=None):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    SN = pd.DataFrame()
    SN['spot_name'] = adata.obs_names

    cropped_images = []
    all_features = []
    img_key = list(adata.uns['spatial'].keys())[0]

    if scale is None:
        scale = adata.uns['spatial'][img_key]['scalefactors']['tissue_hires_scalef']
    diameter = adata.uns['spatial'][img_key]['scalefactors']['spot_diameter_fullres']

    win = diameter * scale
    
    for spot_scale in spot_scale:
        scale_features = []
        for idx, row in SN.iterrows():
            y, x = int(adata.obsm['spatial'][idx][1] * scale), int(adata.obsm['spatial'][idx][0] * scale)
            radius = int(round(win // 2 * spot_scale))
            cropped_image = image[y - radius:y + radius, x - radius:x + radius]
        
            cropped_images.append(cropped_image)
            
            # Calculate the histogram features
            hist_features = features_histogram(
                cropped_image,
                layer='image_array',
                library_id=None,
                spot_scale=spot_scale,
                feature_name="histogram",
                channels=[0, 1, 2],  # R, G, B channels
                bins=bins,
                v_range=(0, 256)
            )
         
            # Flatten and store the features
            flattened_features = flatten_features(hist_features, spot_scale)
            scale_features.append(flattened_features)
        
        # Convert the list of feature dictionaries for this scale into a DataFrame
        scale_df = pd.DataFrame(scale_features)
        all_features.append(scale_df)

    # Concatenate all the features DataFrames horizontally
    features_df = pd.concat(all_features, axis=1)
    features_df.index = adata.obs_names
    adata.obsm['features'] = features_df
    adata.obsm['features'] = adata.obsm['features'].loc[:, (adata.obsm['features'] != 0).any(axis=0)]

    scaler = MinMaxScaler()
    scaled=scaler.fit(adata.obsm['features'])
    adata.obsm['features'] = scaled.transform(adata.obsm['features'])

    return features_df





def extract_features_visiumhd(sdata, img_path, json_path, spot_scale: list, bins: int = 10, resolution:str = 'square_016um',scale=None):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(json_path, 'r') as file:
        jsondata = json.load(file) 

    SN = pd.DataFrame()
    SN['spot_name'] = sdata.tables[resolution].obs_names

    cropped_images = []
    all_features = []


    if scale is None:
        scale = jsondata.get('tissue_hires_scalef')
    diameter = jsondata.get('spot_diameter_fullres')

    win = diameter * scale
    
    for spot_scale in spot_scale:
        scale_features = []
        for idx, row in SN.iterrows():
            y, x = int(sdata.tables[resolution].obsm['spatial'][idx][1] * scale), int(sdata.tables[resolution].obsm['spatial'][idx][0] * scale)
            radius = int(round(win // 2 * spot_scale))
            cropped_image = image[y - radius:y + radius, x - radius:x + radius]
        
            cropped_images.append(cropped_image)
            
            # Calculate the histogram features
            hist_features = features_histogram(
                cropped_image,
                layer='image_array',
                library_id=None,
                spot_scale=spot_scale,
                feature_name="histogram",
                channels=[0, 1, 2],  # R, G, B channels
                bins=bins,
                v_range=(0, 256)
            )
         
            # Flatten and store the features
            flattened_features = flatten_features(hist_features, spot_scale)
            scale_features.append(flattened_features)
        
        # Convert the list of feature dictionaries for this scale into a DataFrame
        scale_df = pd.DataFrame(scale_features)
        all_features.append(scale_df)

    # Concatenate all the features DataFrames horizontally
    features_df = pd.concat(all_features, axis=1)
    features_df.index = sdata[resolution].obs_names
    sdata.tables[resolution].obsm['features'] = features_df
    sdata.tables[resolution].obsm['features'] = sdata.tables[resolution].obsm['features'].loc[:, (sdata.tables[resolution].obsm['features'] != 0).any(axis=0)]

    scaler = MinMaxScaler()
    scaled=scaler.fit(sdata.tables[resolution].obsm['features'])
    sdata.tables[resolution].obsm['features'] = scaled.transform(sdata.tables[resolution].obsm['features'])

    return features_df
