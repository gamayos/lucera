# CLC2018 color palette and class codes from dataset documentation:
clc_palette = [
  '9e9e9e','dcdcdc','cc99ff','cc66ff','9900ff','ffcc99','ff6600','ff0000','cc3300','ffff00',
  'ccff33','99ff33','66ff00','33cc00','009900','006600','99cc00','cccc00','669900','99ffcc',
  '66cc99','339966','006633','99ffff','66ffff','33cccc','009999','003366','99ccff','6699ff',
  '3366ff','0033cc','000099','9999ff','6666cc','333399','ff99cc','ff66cc','ff3399','cc0066',
  '990033','ffcccc','ff99cc','ff66cc'
]

# Class values 1–44 (sequential order per docs)
clc_vis = {'min': 1, 'max': 44, 'palette': clc_palette}

import ipywidgets as widgets
layout = widgets.Layout(
    overflow='auto',
    border='1px solid gray',
    width='250px',
    height='350px',        # 👈 adjust height here
    padding='5px'
)

legend_dict = {
    "111 Continuous urban fabric": "e6004d",
    "112 Discontinuous urban fabric": "ff0000",

    "121 Industrial or commercial units": "cc4df2",
    "122 Road and rail networks and associated land": "cc0000",
    "123 Port areas": "e6cccc",
    "124 Airports": "e6cce6",

    "131 Mineral extraction sites": "a600cc",
    "132 Dump sites": "a64d00",
    "133 Construction sites": "ff4dff",

    "141 Green urban areas": "ffa6ff",
    "142 Sport and leisure facilities": "ffe6ff",

    "211 Non-irrigated arable land": "ffffa8",
    "212 Permanently irrigated land": "ffff00",
    "213 Rice fields": "e6e600",

    "221 Vineyards": "e68000",
    "222 Fruit trees and berry plantations": "f2a64d",
    "223 Olive groves": "e6a600",

    "231 Pastures": "e6e64d",

    "241 Annual crops associated with permanent crops": "ffe6a6",
    "242 Complex cultivation patterns": "ffe64d",
    "243 Land principally occupied by agriculture, with significant areas of natural vegetation": "e6cc4d",
    "244 Agro-forestry areas": "f2cca6",

    "311 Broad-leaved forest": "80ff00",
    "312 Coniferous forest": "00a600",
    "313 Mixed forest": "4dff00",

    "321 Natural grasslands": "ccf24d",
    "322 Moors and heathland": "a6ff80",
    "323 Sclerophyllous vegetation": "a6e64d",
    "324 Transitional woodland-shrub": "a6f200",

    "331 Beaches, dunes, sands": "e6e6e6",
    "332 Bare rocks": "cccccc",
    "333 Sparsely vegetated areas": "ccffcc",
    "334 Burnt areas": "000000",
    "335 Glaciers and perpetual snow": "a6e6cc",

    "411 Inland marshes": "a6a6ff",
    "412 Peat bogs": "4d4dff",

    "421 Salt marshes": "ccccff",
    "422 Salines": "e6e6ff",
    "423 Intertidal flats": "a6a6e6",

    "511 Water courses": "00ccf2",
    "512 Water bodies": "80f2e6",

    "521 Coastal lagoons": "00ffa6",
    "522 Estuaries": "a6ffe6",
    "523 Sea and ocean": "e6f2ff"
}

codes = [int(k[:3]) for k in legend_dict.keys()]

vis = {'min': 1, 'max': len(codes), 'palette': list(legend_dict.values())}

