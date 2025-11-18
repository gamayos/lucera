import ipywidgets as widgets
import importlib, h3, ee, geemap
import lucera.viz.corine as corine
import lucera.data.datautils as du

importlib.reload(corine)
importlib.reload(du)

def init_gee_map(region='Podkarpackie',zoom=8):

    region = (ee.FeatureCollection('FAO/GAUL/2015/level1')
                    .filter(ee.Filter.eq('ADM1_NAME', region))
                    .geometry())

    #region = 'Podkarpackie' #'Malopolske'
    #data = du.init_data(region)
    #data.corine_legend_dict = corine.legend_dict

    # Visualize (pseudo-RGB from embedding bands)
    m = geemap.Map()
    m.centerObject(region, zoom)
    #m.addLayer(data.emb, {'min':-0.3,'max':0.3,'bands':['A01','A16','A09']}, 'Embeddings 2018')

    # Remaps CORINE class codes (111…523) to sequential indices (1…44).
    #clc_idx = data.clc.remap(corine.codes, list(range(1, len(corine.codes)+1))).rename('landcover_idx')

    # Adds the CORINE raster to the map, colored according to EEA’s palette
    #m.addLayer(clc_idx, corine.vis, 'CORINE 2018')

    # Creates a scrollable legend showing all 44 class names with their corresponding colors.
    #legend = m.add_legend(title="CORINE Land Cover 2018", legend_dict=corine.legend_dict)

    #poly = h3.LatLngPoly(data.region.getInfo()['coordinates'][0])
    #cells4 = h3.h3shape_to_cells_experimental(poly, res=4, contain='overlap')
    #cells6 = du.get_cell_children_in_region(cells4[0], data.region.getInfo(), res=6)

    #features = [du.cell_to_feature(h) for h in sorted(cells4)]
    #hex_fc = {"type": "FeatureCollection", "features": features}

    #m.add_geojson(
    #    hex_fc,
    #    layer_name="H3 boundary res4",
    #    style={"color": "#ff7f0e", "fillColor": "#ff7f0e", "fillOpacity": 0.1, "weight": 2},
    #)

    #features = [du.cell_to_feature(h) for h in sorted(cells6)]
    #hex_fc = {"type": "FeatureCollection", "features": features}

    #m.add_geojson(
    #    hex_fc,
    #    layer_name="H3 boundary res6",
    #    style={"color": "#ff7f0e", "fillColor": "#ff7f0e", "fillOpacity": 0.1, "weight": 2},
    #)

    return m