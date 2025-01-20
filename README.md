# Bathymetry Data Inpainting
This repository contains the code used to implement inpainting on bathymetry data. The project is composed of two main parts: Preprocessing and Inpainting.
Composed mainly of two parts: preprocessing of the data and inpainting.

## Table of contents
- [Preprocessing](#Preprocessing)
- [Inpainting](#Inpainting)
- [Requirements](#Requirements)
- [Usage](#Usage)

## Preprocessing
This phase is used to transforme the raw data, downloaded from EMODnet and Copernicus Land, into the 3D NumPy ndarrays stored in `data/Complete/filled/`. Those arrays represent small portions of a larger map and are made as follows:
- `array[0, :, :]` - longitudes
- `array[1, :, :]` - latitudes
- `array[2, :, :]` - depths (negative values) and heights (positive values). Where the data is unknown, the value is `np.nan`.
To create these arrays, we must download the raw data from EMODnet and Copernicus land and convert them into a more suitable format.

### Preprocessing bathymetry data
It can be done wit the `run_bathymetry.mk` makefile, that has the following phony targets:
- `downoad`: downloads the data corresponding to the tiles [E5](https://emodnet.ec.europa.eu/geonetwork/emodnet/eng/catalog.search#/metadata/dae64d8b-4a7e-4391-a531-e2c66ac8a489), [E6](https://emodnet.ec.europa.eu/geonetwork/emodnet/eng/catalog.search#/metadata/b1b7e023-db0c-4b71-a8db-102f8687f521), [F5](https://emodnet.ec.europa.eu/geonetwork/emodnet/eng/catalog.search#/metadata/11a175e2-841f-4c23-9592-4ec623f563e7) and [F6](https://emodnet.ec.europa.eu/geonetwork/emodnet/eng/catalog.search#/metadata/9fc5b391-086c-41b2-99d6-8c8f0ecedfff) from the [Emodnet Bathymetry Website](https://emodnet.ec.europa.eu/geonetwork/emodnet/eng/catalog.search#/search?any=Bathymetry%20and%20Elevation) and stores them in the `data/Bathymetry/EMODnet_bathymetry/raw/`

- `run`: transforms those data into smaller images of size `width` $\times$ `height`, containing only the available sea depth data; the resulting image are in the form of `numpy` 3D darrays, stored in the `data/Bathymetry/EMODnet_bathymetry/cutted/` folder. A zipped copy of those images, that is more suitable to be uploaded to GitHub, is saved into `data/Bathymetry/EMODnet_bathymetry/cutted_zipped/`
- `unzip`: takes the zipped bathymetry arrays from `data/Bathymetry/EMODnet_bathymetry/cutted_zipped/` and unzips them to `data/Bathymetry/EMODnet_bathymetry/cutted/`.
- `clean_download`: remove the `data/Bathymetry/EMODnet_bathymetry/raw/` folder with all its content.

### Preprocessing land data

It can be done using the `run_land.mk` makefile, with the following phony:

- `download`: download the data from the [Copernicus Land Website](https://egms.land.copernicus.eu/) and stores it into ``data/Land/Copernicus_land_ortho/raw`. The downloaded tiles are the ones in the square `E34N13`, `E34N34` `E56N13`, `E56N34`. Since the website requires the user to login before downloading the data, the `id` must be manually inserted in the makefile.
- `run`: selects only the useful information from the downloaded data and saves it in the `data/Land/Copernicus_land_ortho/converted` folder as `csv` files. A zipped copy of those files, that is more suitable to be uploaded to GitHub, is saved into `data/Land/Copernicus_land_ortho/converted_zipped/`
- `unzip`: takes the zipped bathymetry arrays from ``data/Land/Copernicus_land_ortho/converted_zipped/` and unzips them to `data/Land/Copernicus_land_ortho/converted`.
- `clean_download`: remove the `data/Land/Copernicus_land_ortho/raw` folder with all its content.

### Preprocessing coastline data

Coastline data is used to understand which bathymetry data are missing. I downloaded it from [EMODnet_Bathymetry_2022_coastlines](https://emodnet.ec.europa.eu/geonetwork/emodnet/eng/catalog.search#/search?resultType=details&sortBy=sortDate&any=EMODnet%20Bathymetry%20-%20World%20Coastline%20version%202022&fast=index&_content_type=json&from=1&to=20). The file contains 3 kinds of coastline:
- LAT (Lowest Astronomical Tide)
- MSL (Mean-Sea-Level)
- MHW (Mean-High-Water)

The preprocessing can be made using the `run_coastline.mk` makefile, with the following phony:
- `download`: download the dataset to `data/Coastlines/EMODnet/raw`.
- `run`: extract the coastline data, choosing between LAT, MSL and MHW, and save it to `data/Coastlines/EMODnet/converted/`.
- `zip`: save the files of `data/Coastlines/EMODnet/converted/` as zipped files into `data/Coastlines/EMODnet/converted_zipped/`, choosing the maximum size of each group.
- `unzip`: unzip the files of `data/Coastlines/EMODnet/converted_zipped/` to `data/Coastlines/EMODnet/converted/`.

### Regridding


## Inpainting

## Requirements
Make sure you have the following dependencies installed:
- `Python 3.11`
- `Numpy < 2.0`
- `netCDF4`

To install all the required packages, run

`pip install src/requirements.txt`

## Usage
To perform preprocessing tasks, run

`make -f <makefile name> <phony target>`
