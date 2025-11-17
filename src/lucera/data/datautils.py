from h3._cy import cell_to_children_size
import numpy as np
import ee, geemap
from types import SimpleNamespace

def list_country_regions(country, v=False):

    fc = (ee.FeatureCollection('FAO/GAUL/2015/level1')
        .filter(ee.Filter.eq("ADM0_NAME", country)))

    # Extract the ADM1_NAME column for all features, convert to list
    adm1_list = fc.aggregate_array('ADM1_NAME').distinct().sort()

    adm1_names = adm1_list.getInfo()
    if v:
        print("Number of regions:", len(adm1_names))
        for name in adm1_names:
            print(name)

    # Bring to Python
    return adm1_names

def init_data(region):

    data = SimpleNamespace()

    data.region = (ee.FeatureCollection('FAO/GAUL/2015/level1')
                    .filter(ee.Filter.eq('ADM1_NAME', region))
                    .geometry())

    data.emb = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                    .filterDate('2021-01-01','2022-01-01')
                    .filterBounds(data.region)
                    .mosaic()
                    .clip(data.region))

    # Loads CORINE Land Cover 2018 raster (44 classes, 100 m resolution).
    data.clc = ee.Image('COPERNICUS/CORINE/V20/100m/2018').select('landcover').clip(data.region)

    return data

def scan_country_regions(data, country, regions=None, v=False):

    if not regions or regions=='all':
        regions = list_country_regions(country)
    
    for n, region in enumerate(regions):
        print(f'Processing {region}, {country} ...')
        try:
            scan_region_for_samples(data, region)
        except:
            print('Failed.')
        print()
        #if n>2: break

import random
def scan_region_for_samples(data, region, v=False):

    region_geo = (ee.FeatureCollection('FAO/GAUL/2015/level1')
                        .filter(ee.Filter.eq('ADM1_NAME', region))
                        .geometry())

    emb = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
       .filterDate('2018-01-01','2019-01-01')
       .filterBounds(region_geo)
       .mosaic())

    poly = h3.LatLngPoly(region_geo.getInfo()['coordinates'][0])

    cells4 = h3.h3shape_to_cells_experimental(poly, res=4, contain='overlap')
    if len(cells4)>16: cells4 = random.sample(cells4, 16)

    dataset = {l: np.zeros((0,64)) for l in data.labels}
    for n4, cell4 in enumerate(cells4):
        cell4dataset = {l: np.zeros((0,64)) for l in data.labels}
        cells6 = get_cell_children_in_region(cell4, region_geo.getInfo(), res=6)
        
        print(cell4,end=' ')
        for n6, cell in enumerate(cells6):
            aoi = cell_to_feature(cell)
            X, Y = extract_samples(emb, ee.Feature(aoi['geometry']))
            cell_labels = [str(y) for y in np.unique(Y)]
            for l in cell_labels:
                try:
                    x = X[Y==int(l)]
                    cell4dataset[l] = np.concat((cell4dataset[l], x), axis=0)
                except:
                    pass
            print('.',end='')
            #if n6>2: break
        print()

        for l, d in cell4dataset.items():
            if  d.shape[0]>data.N:
                d = d[np.random.choice(d.shape[0], size=data.N, replace=False)]
            dataset[l] = np.concat((dataset[l], d), axis=0)

        #if n4>2: break

    for l, d in dataset.items():
        if  d.shape[0]>data.N:
            dataset[l] = d[np.random.choice(d.shape[0], size=data.N, replace=False)]
        if v: print(l, d.shape)

    data.regional[region] = dataset

def extract_samples(emb, aoi):

    aoiemb = emb.clip(aoi)

    # Export or sample
    samples = aoiemb.sample(
        region=aoi.geometry(),
        scale=100,
        projection='EPSG:4326',
        numPixels=5000,    # adjust depending on area
        seed=0,
        geometries=False
    )

    #band_names = aoiemb.bandNames().getInfo()
    #print(band_names)

    df = geemap.ee_to_df(samples)
    X = df[[f'A{i:02}' for i in range(64)]].to_numpy()

    #print(X.shape)   # (n, 64)

    # Loads CORINE Land Cover 2018 raster (44 classes, 100 m resolution).
    aoiclc = ee.Image('COPERNICUS/CORINE/V20/100m/2018').select('landcover').clip(aoi)

    # Export or sample
    labels = aoiclc.sample(
        region=aoi.geometry(),
        scale=100,
        projection='EPSG:4326',
        numPixels=5000,    # adjust depending on area
        seed=0,
        geometries=False
    )

    band_names = aoiclc.bandNames().getInfo()
    #print(band_names)

    df = geemap.ee_to_df(labels)
    Y = df['landcover'].to_numpy()

    return X, Y

def dataset_from_regional_collection(datafile, labels, N=1000, v=False):

    loaded = np.load(datafile, allow_pickle=True)
    dataset = loaded["data"].item()

    if v: print(dataset.regional.keys())

    DD = {l: np.zeros((0,64)) for l in labels}
    for rdata in dataset.regional.values():
        for l, d in rdata.items():
            if  d.shape[0]>N:
                d = d[np.random.choice(d.shape[0], size=N, replace=False)]
            DD[l] = np.concatenate((DD[l], d), axis=0)


    XX = np.zeros((0,64))
    YY = []
    for l, d in DD.items():
        if  d.shape[0]>N:
            DD[l] = d[np.random.choice(d.shape[0], size=N, replace=False)]
        
    DD = {l:d for l,d in DD.items() if d.shape[0]>0}

    for l, d in DD.items():
        if v: print(l, d.shape)
        XX = np.concatenate((XX, d), axis=0)
        YY = YY + [int(l)]*d.shape[0]

    return XX, YY

def get_cells_from_feature(region, res):

    poly = h3.LatLngPoly(region.getInfo()['coordinates'][0])
    cells = h3.h3shape_to_cells_experimental(poly, res=res, contain='overlap')

    return cells

def get_cell_children_in_region(cell, region, res):

    overlap = get_cell_polygon_overlap(cell, region)

    geom = overlap["geometry"]
    cells = set()

    if geom["type"] == "Polygon":
        poly = h3.LatLngPoly(geom["coordinates"][0])
        cells |= set(h3.h3shape_to_cells_experimental(poly, res=6, contain='overlap'))

    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            poly = h3.LatLngPoly(poly[0])
            cells |= set(h3.h3shape_to_cells_experimental(poly, res=6, contain='overlap'))

    hex7s = list(h3.cell_to_children(cell, res))
    cells = [cell for cell in cells if cell in hex7s]

    return cells

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

def get_outliers(X, method='OCSVM', contam=0.5):

    if method=='OCSVM':
        oc = OneClassSVM(kernel="rbf", gamma="scale", nu=contam)
        oc.fit(X)
        y_hat = oc.predict(X)                  # +1 = inlier, -1 = outlier
        scores = oc.decision_function(X)

    elif method=='LOF':
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contam)
        y_hat = lof.fit_predict(X)
        #n_errors = (y_pred != ground_truth).sum()
        scores = lof.negative_outlier_factor_

    elif method == 'RCOV':
        # Robust covariance / Elliptic Envelope (Minimum Covariance Determinant)
        rcov = EllipticEnvelope(contamination=contam, support_fraction=None)
        rcov.fit(X)
        y_hat = rcov.predict(X)                # +1 = inlier, -1 = outlier
        scores = rcov.decision_function(X)     # higher = more normal

    else:
        raise ValueError(f"Unknown method: {method}")

    inliers = (y_hat == 1)
    outliers = (y_hat == -1)

    return inliers, outliers, scores

def reduce_dim(X, method='UMAP'):

    if method=='UMAP':
        reducer = umap.UMAP(
            n_neighbors=15,    # balance local/global structure
            min_dist=0.1,      # controls cluster tightness
            n_components=2,    # 2D for visualization
            random_state=42,
            metric='cosine'
        )
    elif method=='TSNE':
        reducer = TSNE(
            n_components=2,
            perplexity=30,     # typical range: 5–50
            learning_rate='auto',
            init='pca',
            random_state=42,
            metric='cosine'
        )
    else: reducer = PCA(n_components=2, random_state=42)

    return reducer, reducer.fit_transform(X)

import h3

def cell_to_feature(h):
    boundary = h3.cell_to_boundary(h)
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [boundary]},
        "properties": {"h3": h, "res": h3.get_resolution(h)},
    }

from shapely.geometry import Polygon, shape, mapping

def cell_overlaps_polygon(h, polygon_geojson):
    """
    Returns True if the H3 hex (string index) overlaps/intersects the polygon.
    polygon_geojson: GeoJSON-like dict or shapely geometry.
    """
    # Convert the H3 cell to a Shapely polygon
    boundary = h3.cell_to_boundary(h)  # [[lat, lon], ...]
    hex_poly = Polygon(boundary)

    # Convert the polygon input
    if not isinstance(polygon_geojson, Polygon):
        poly = shape(polygon_geojson)
    else:
        poly = polygon_geojson

    return hex_poly.intersects(poly)

def get_cell_polygon_overlap(h3_cell: str, aoi_geojson: dict):
    """
    Returns a GeoJSON geometry (Polygon/MultiPolygon) of the overlap between
    the given H3 cell and AOI polygon. Returns None if there is no overlap.
    """
    # H3 boundary comes as [[lat, lon], ...] → convert to [lon, lat]
    latlon = h3.cell_to_boundary(h3_cell)
    ring = list(latlon) #[(lon, lat) for lat, lon in latlon]
    if ring[0] != ring[-1]:
        ring.append(ring[0])  # close ring

    hex_poly = Polygon(ring)
    aoi_poly = shape(aoi_geojson)  # aoi_geojson can be Polygon/MultiPolygon

    inter = hex_poly.intersection(aoi_poly)
    if inter.is_empty:
        return None

    geom = mapping(inter)

    # Return a GeoJSON geometry (Polygon or MultiPolygon)
    return {"type":"Feature","geometry":geom,"properties":{}}

import numpy as np
import pandas as pd

def match_clusters_to_classes(Y_cluster, Y_corine):
    """
    Returns:
      mapping: dict {cluster_label -> dominant CORINE class}
      table: pd.DataFrame contingency table
    """
    Y_cluster = np.asarray(Y_cluster)
    Y_corine = np.asarray(Y_corine)
    
    # Build contingency table
    df = pd.DataFrame({
        "cluster": Y_cluster,
        "corine": Y_corine
    })

    contingency = pd.crosstab(df["cluster"], df["corine"])

    # Dominant class for each cluster
    mapping = contingency.idxmax(axis=1).to_dict()

    return mapping, contingency


# Example usage:
# mapping, table = match_clusters_to_classes(Y_cluster, Y_corine)
# print(mapping)
# print(table)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

def train_classifier(X,  Y):
    # X: embeddings (n_samples, n_features)
    # Y: distilled labels (n_samples,)

    # 1) Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    # 2) Build RBF SVM pipeline (with scaling)
    rbf_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,        # regularization strength
            gamma="scale",# RBF width; "scale" is usually good
            probability=False,  # True if you need predict_proba
            class_weight=None   # or "balanced" for imbalanced data
        ))
    ])

    # 3) Fit model
    rbf_svm.fit(X_train, y_train)

    # 4) Evaluate
    y_pred = rbf_svm.predict(X_test)

    return rbf_svm, y_test, y_pred

