import numpy as np
import rasterio
from rasterio.transform import from_origin

def labels_to_clc_geotiff(
    gdf,
    labels,
    legend_dict,
    out_path,
    decimals=6
):
    """
    Convert classified pixel labels + coordinates into a georeferenced
    1-band GeoTIFF with a CORINE color table.

    Parameters
    ----------
    lons, lats : 1D arrays of length N
        Longitudes and latitudes of pixel centers (EPSG:4326).
        Assumed to form a regular grid.
    labels : 1D array of length N
        Classified CORINE codes (e.g. 111, 211, ...).
    legend_dict : dict
        e.g. {"111 Continuous urban fabric": "e6004d", ...}
    out_path : str
        Output GeoTIFF filename.
    decimals : int
        Number of decimals to round coords when inferring grid.
    """

    lons = gdf.geometry.x.to_numpy()
    lats = gdf.geometry.y.to_numpy()

    lons = np.asarray(lons)
    lats = np.asarray(lats)
    labels = np.asarray(labels)

    # --- 1) Build mapping from CORINE code -> hex color
    code_to_color_hex = {}
    for k, v in legend_dict.items():
        code_str = k.split()[0]  # "111" from "111 Continuous urban fabric"
        # ensure leading '#'
        hex_color = v if v.startswith("#") else f"#{v}"
        code_to_color_hex[code_str] = hex_color

    # --- 2) Normalize coordinates (round to avoid float noise)
    lons_r = np.round(lons, decimals=decimals)
    lats_r = np.round(lats, decimals=decimals)

    # --- 3) Infer regular grid from unique lon/lat
    xs = np.unique(lons_r)
    ys = np.unique(lats_r)

    # sort such that row 0 is max latitude (north), last row is min latitude (south)
    xs = np.sort(xs)
    ys = np.sort(ys)[::-1]

    width = len(xs)
    height = len(ys)

    # Build dicts for fast coordinate -> index lookup
    x_to_col = {x: j for j, x in enumerate(xs)}
    y_to_row = {y: i for i, y in enumerate(ys)}

    # --- 4) Map CORINE labels to integer class IDs
    unique_codes = sorted({str(c) for c in labels})
    code_to_id = {code: i for i, code in enumerate(unique_codes)}

    # Initialize raster with nodata
    nodata = 255
    raster = np.full((height, width), nodata, dtype=np.uint8)

    # Fill raster
    for lon_val, lat_val, lab in zip(lons_r, lats_r, labels):
        if lon_val not in x_to_col or lat_val not in y_to_row:
            continue  # skip if somehow outside grid
        row = y_to_row[lat_val]
        col = x_to_col[lon_val]
        code_str = str(lab)
        class_id = code_to_id.get(code_str, nodata)
        if class_id == nodata:
            # unknown code, leave as nodata
            continue
        raster[row, col] = class_id

    # --- 5) Build color table: class_id -> (R, G, B, A)
    colormap = {}
    for code, class_id in code_to_id.items():
        hex_color = code_to_color_hex.get(code, "#999999")
        # hex to RGB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        colormap[class_id] = (r, g, b, 255)

    # nodata color (optional: transparent)
    # colormap[nodata] = (0, 0, 0, 0)

    # --- 6) Define geotransform (top-left corner, pixel size)
    west = xs.min()
    east = xs.max()
    north = ys.max()
    south = ys.min()

    # derive pixel size from coords
    if len(xs) > 1:
        pixel_size_x = float(np.mean(np.diff(xs))) 
    else:
        pixel_size_x = 0.0001  # fallback

    if len(ys) > 1:
        pixel_size_y = float(abs(np.mean(np.diff(ys))))  # use ABS here
    else:
        pixel_size_y = 0.0001

    transform = from_origin(west, north, pixel_size_x, pixel_size_y)

    # --- 7) Write GeoTIFF with rasterio
    crs = "EPSG:4326"

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=raster.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(raster, 1)
        dst.write_colormap(1, colormap)

    print(f"GeoTIFF written to {out_path}")
