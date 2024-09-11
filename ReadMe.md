# ISO Clustering for Multispectral Image Classification

This repository contains a Python script that performs ISO (Iterative Self-Organizing) clustering on a multispectral image. The script uses Principal Component Analysis (PCA) for dimensionality reduction before applying clustering. The result is a classified raster image saved as a GeoTIFF file.

## Requirements
  To run this script, you'll need the following Python libraries:

  - numpy
  - osgeo (GDAL)
  - sklearn
  - matplotlib
  - You can install the required packages using pip:
    pip install numpy gdal scikit-learn matplotlib

## Script Description
Functions
ISO(X, max_iter=15, n_clusters=15): This function implements the ISO clustering algorithm. It normalizes the data, randomly initializes centroids, and iteratively updates the centroids based on the assigned clusters.

Workflow
  - Load the Multispectral Image: The script uses GDAL to load a multispectral image from the specified path.
  - Flatten the Image Data: The image data is reshaped into a 2D array where each pixel is represented as a row.
  - Perform PCA: The script reduces the dimensionality of the data using PCA, retaining a specified number of components.
  - Apply ISO Clustering: The ISO function is applied to the PCA-transformed data to classify the image into clusters.
  - Save the Classified Image: The classified image is saved as a GeoTIFF file, preserving the original geospatial metadata.

Parameters
  - image_path: Path to the input multispectral image file.
  - n_components: Number of principal components to retain in PCA.
  - n_clusters: Number of clusters for ISO clustering.
  - max_iter: Maximum number of iterations for the ISO clustering algorithm.
  - classified_image_path: Path to save the classified raster image.

### Output
The script generates a classified raster image saved as a GeoTIFF file, which contains cluster labels corresponding to different regions in the original image.
