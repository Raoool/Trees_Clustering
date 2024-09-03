import numpy as np
from osgeo import gdal, gdal_array
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def ISO(X, max_iter=15, n_clusters=15):
    # Normalizing the data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Initializing cluster centroids randomly
    centroids = np.random.rand(n_clusters, X_normalized.shape[1])
    
    # Iterate
    for _ in range(max_iter):
        # Assigning points to clusters based on nearest centroid
        distances = np.linalg.norm(X_normalized[:, np.newaxis, :] - centroids, axis=-1)
        cluster_labels = np.argmin(distances, axis=1)
        
        # Updating cluster centroids
        for i in range(n_clusters):
            cluster_points = X_normalized[cluster_labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
    
    return cluster_labels

# Loading masked image
image_path = 'masked_image_April_12th.tif'
dataset = gdal.Open(image_path)
bands = [dataset.GetRasterBand(i).ReadAsArray() for i in range(1, dataset.RasterCount + 1)]
data = np.stack(bands, axis=-1)

# Flattening the data
rows, cols, bands = data.shape
data_2d = data.reshape(rows * cols, bands)

# Performing PCA
n_components = 4  
pca = PCA(n_components=n_components)
pca.fit(data_2d)
X_pca = pca.transform(data_2d)

# number of clusters (K)
n_clusters = 15

# Applying ISO clustering
cluster_labels = ISO(X_pca, max_iter=10, n_clusters=n_clusters)

# Reshaping cluster labels to the original image dimensions
cluster_labels_image = cluster_labels.reshape(rows, cols)

# Creating a new GeoTIFF file to save the classified raster image
classified_image_path = '/ISO_classified_image_april11th_2ndrun.tif'

# Defining metadata for the new GeoTIFF file
driver = gdal.GetDriverByName('GTiff')
output_dataset = driver.Create(classified_image_path, cols, rows, 1, gdal.GDT_Float32)

# Copying geotransform and projection from the original image
output_dataset.SetGeoTransform(dataset.GetGeoTransform())
output_dataset.SetProjection(dataset.GetProjection())

# Writing the classified raster image to the new GeoTIFF file
output_band = output_dataset.GetRasterBand(1)
output_band.WriteArray(cluster_labels_image)


output_dataset = None

print("Classified raster image saved successfully as", classified_image_path)

