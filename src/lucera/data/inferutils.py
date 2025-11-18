import ee, geemap, joblib, importlib
import lucera.viz.geoutils as gu
importlib.reload(gu)

def infer_from_aoi(aoi_feature,
    model_path='/content/lucera/models/clc-classifier-poland-2018-20251119.joblib',
    year='2018', 
    scale=100):

    aoi_geom = aoi_feature.geometry()

    # AlphaEarth embeddings
    emb_ic = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
            .filterDate(f'{year}-01-01', f'{int(year)+1}-01-01')
            .filterBounds(aoi_geom))
    emb_img = emb_ic.mosaic().clip(aoi_geom)

    # Ensure we have band names embedding_0..embedding_63
    bands = emb_img.bandNames().getInfo()
    #if len(bands) == 64 and bands[0] == 'b1':
    #    emb_img = emb_img.rename([f"embedding_{i}" for i in range(64)])

    # Sample WITH geometries
    scale = scale
    samples = emb_img.sample(
        region=aoi_geom,
        scale=scale,
        numPixels=1e13,
        geometries=True
    )

    # Convert to GeoDataFrame so we keep geometry explicitly
    gdf = geemap.ee_to_gdf(samples)

    # Load trained CORINE classifier (pipeline)
    clf = joblib.load(model_path)

    feature_cols = [f"A{i:02}" for i in range(64)]
    X = gdf[feature_cols].to_numpy()

    # 2. Predict classes
    y_pred = clf.predict(X)

    # 3. Attach predictions to DataFrame
    gdf["clc"] = y_pred

    return gdf

