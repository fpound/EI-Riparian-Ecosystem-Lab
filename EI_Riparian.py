#=============================================================================#
#                          Import Packages                                    #
#=============================================================================#
import fiona, os, glob, rasterio, requests, shutil

import numpy as np
import pandas as pd

from pysheds.grid import Grid
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import mapping, shape, Polygon
from joblib import Parallel, delayed
from retry import retry
from tqdm import tqdm

import ee
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')




DataFolder = '/Volumes/Lightning Strike/EARTHSHOT_DATA/BRAT/Idaho/_NEW_DATA'
InputData = '/Volumes/Lightning Strike/EARTHSHOT_DATA/BRAT/Idaho/Dams/02_Dams_Virtual_Census/Shapefiles/Beaver_Dams/Idaho_BeaverDams_Idaho.shp'


#=============================================================================#
#                          Define Fuctions                                    #
#=============================================================================#



# Returns a Geodataframe from either .tif or .shp.
# This is then used in otehr functions which defines a bounding box etc. 
def makeExtent(InputData):

    if InputData.endswith('.shp'):
        extent = makeGDF(InputData)

    elif InputData.endswith('.tif'):
        extent = makeExtentPoly(InputData)

    else:
        print("Data type not known. Must be a .shp or .tif")
    return extent



# Returns GDF from .shp
def makeGDF(shape_path):
    shape_gdf = gpd.read_file(shape_path)
    return shape_gdf



# Returns GDF from .tif
def makeExtentPoly(raster_path):
    
    raster = rasterio.open(raster_path)
    raster_name = os.path.split(InputData)[-1].split('.')[0]
    
    left = raster.bounds[0] 
    lower = raster.bounds[1]
    right = raster.bounds[2]
    upper = raster.bounds[3]

    # Set raster bounding box
    raster_box = [[left, upper], [right, upper], [right, lower], [left, lower]]

    # define the geometry
    raster_poly = Polygon(raster_box)
    d = {'name':[f'{raster_name}',], 'geometry': [raster_poly,]}

    # make the geodataframe
    raster_extent = gpd.GeoDataFrame(d, crs=str(raster.crs))
    
    return raster_extent





## This is a key function which downloads a DEM from the USGS National Map.

def Download_DEM(path, Newest=True):
    
    geodataframe = makeExtent(path)
    geodataframe = geodataframe.to_crs("EPSG:4269")
    
    
    xMin = round(geodataframe.total_bounds[0], 3)
    yMin = round(geodataframe.total_bounds[1], 3)
    xMax = round(geodataframe.total_bounds[2], 3)
    yMax = round(geodataframe.total_bounds[3], 3)
    boundingBox = "{},{},{},{}".format(xMin, yMin, xMax, yMax)


    linkStart = "https://tnmaccess.nationalmap.gov/api/v1/products?datasets=National%20Elevation%20Dataset%20(NED)%201/3%20arc-second&"
    boundingBox_forLink = "bbox=" + str(boundingBox)
    linkFinish = "&prodFormats=GeoTIFF&outputFormat=JSON"
    TNM_Link = linkStart + boundingBox_forLink + linkFinish

    r = requests.get(TNM_Link)
    json_data = r.json()
    downloadList = []

    for item in json_data["items"]:
        downloadList.append(item["downloadURL"])

    
    # Defaults to only downloading the newest DEMs. If all available DEMs are desired, set Newest=False
    if Newest:
        root_name = []
        for eachFile in downloadList:
            name = os.path.split(eachFile)[0]
            if name not in root_name:
                root_name.append(name)
        DownloadListNewest = []
        for name in root_name:
            newestFile = []
            for link in downloadList:
                if name in link:
                    newestFile.append(link)
            DownloadListNewest.append(max(newestFile))
        downloadList = DownloadListNewest

    fileCounter = 1
    RastersList = []

    for downURL in downloadList:
        fileSplit = downURL.split("/")
        fileName = fileSplit[-1]

        filePath = os.path.join('DEM', fileName)

        RastersList.append(filePath)

        print (f'Downloading {fileName} file {fileCounter} of {len(downloadList)}')

        fileDown = requests.get(downURL)

        with open(filePath, 'wb') as asdf:
            asdf.write(fileDown.content)
        fileCounter += 1

        print ('Finished downloading')

    return RastersList




## Another key function, this processes a DEM using watershed analysis functions
## powered by the Pysheds package. For the purposes of EI Riparian scripts,
## this function returns a shp file path of the stream vector network.

def watershed_preProcessing(dem_path):

    
    DEM_name = dem_path.split('/')[-1].split('.tif')[0]

    
    grid = Grid.from_raster(dem_path, data_name='dem')
    dem = grid.read_raster(dem_path)

    # Detect pits
    pits = grid.detect_pits(dem)

    # Fill pits
    pit_filled_dem = grid.fill_pits(dem)
    pits = grid.detect_pits(pit_filled_dem)
    assert not pits.any()

    # Detect depressions
    depressions = grid.detect_depressions(pit_filled_dem)

    # Fill depressions
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    depressions = grid.detect_depressions(flooded_dem)
    assert not depressions.any()

    # Detect flats
    flats = grid.detect_flats(flooded_dem)

    # Fill flats
    inflated_dem = grid.resolve_flats(flooded_dem)
    flats = grid.detect_flats(inflated_dem)

    # Compute flow direction based on corrected DEM
    fdir = grid.flowdir(inflated_dem)
    
    acc = grid.accumulation(fdir)
    branches = grid.extract_river_network(fdir, acc > 2500)
    
    geojson_path = f'StreamVectors/{DEM_name}_sv.geojson'
    f = open(geojson_path, 'w')
    f.write(str(branches))
    f.close()
    
    streamNet = gpd.read_file(geojson_path)
    shp_path = f'StreamVectors/{DEM_name}_sv.shp'
    streamNet.to_file(shp_path)
    
    #os.remove(geojson_path)
    
    return shp_path




## This function takes the previous stream network lines, 
## and makes points along those streams. The second parameter is the distance
## in meters between each point along the lines.  


def makePointsFromLines(Line_SHP, Distance_Points_M):
    
    DEM_name = os.path.split(Line_SHP)[-1].split('.shp')[0]

    shp_name = os.path.split(Line_SHP)[-1].split('.shp')[0]
    
    SHP = gpd.read_file(Line_SHP)
    
    ## need to reproject for to the points to work correctly
    ## Should come up with a way to identify the epsg programatically
    
    rePro_shp = SHP.to_crs("EPSG:3857")
    rePro_shp_path = f'{shp_name}_repro.shp'
    rePro_shp.to_file(rePro_shp_path)
    
    
    lines = fiona.open(rePro_shp_path)

    # creation of the resulting shapefile
    schema = {'geometry': 'Point','properties': {'id': 'int'}}

    crs = lines.crs
    points_path = f'POINTS/{DEM_name}_Points.shp'

    
    
    with fiona.open(points_path,
                    'w', 'ESRI Shapefile', schema, crs=crs) as output:

        for line in lines:

            geom = shape(line['geometry'])

            # length of the LineString
            length = geom.length

            # create points every x meters along the line
            for i, distance in enumerate(range(0, int(length), Distance_Points_M)):
                point = geom.interpolate(distance)   
                output.write({'geometry':mapping(point),'properties': {'id':i}}) 
    
    
    for filename2 in glob.glob(f"{rePro_shp_path[:-4]}*"):
        os.remove(filename2)
    
    points_gdf = gpd.read_file(points_path)
    points_repro = points_gdf.to_crs("EPSG:4269")
    points_repro['id'] = np.arange(len(points_repro))
    
    return points_repro, points_path





## This function uses Google Earth Engine to download NAIP image chips.
## The dates must be defined in the code itself.  



def points_toNAIP_Chips(points, inputName, DEM_name, optString=""):
    if points.empty:
        return
    #points.to_crs(DEM.crs)
    
    points_gee = []
    for i in range(0, len(points)):
        
        x = float(points.iloc[i].geometry.centroid.x)
        y = float(points.iloc[i].geometry.centroid.y)
        geom = {'geodesic': False,
                'type': 'Point', 
                'coordinates': [x, y]}
        points_gee.append(geom)
    #points_gee

    left = points.total_bounds[0] 
    lower = points.total_bounds[1]
    right = points.total_bounds[2]
    upper = points.total_bounds[3]
    point_box = [[left, upper], [right, upper], [right, lower], [left, lower], [left, upper]]

    region = ee.Geometry.Polygon(
        [
            point_box
        ],
        None,
        False,
    )
    
    
    image = (
    ee.ImageCollection('USDA/NAIP/DOQQ')
    .filterBounds(region)
    .filterDate('2018-01-01', '2019-12-31')  ### need to figure out how to properly define the years
    .mosaic()
    .clip(region)
    .select('R', 'G', 'B', 'N')
    )
    
    
    params = {
    'count': 100,  # How many image chips to export
    'buffer': 127,  # The buffer distance (m) around each point
    'scale': 100,  # The scale to do stratified sampling
    'seed': 1,  # A randomization seed to use for subsampling.
    'dimensions': '256x256',  # The dimension of each image chip
    'format': "GEO_TIFF",  # The output image format, can be png, jpg, ZIPPED_GEO_TIFF, GEO_TIFF, NPY
    'prefix': f'{optString}_chip_{DEM_name}',  # The filename prefix
    'processes': 25,  # How many processes to used for parallel processing
    'out_dir': f'NAIP/CHIPS_{inputName}',  # The output directory. Default to the current working directly
    }

    
    @retry(tries=10, delay=1, backoff=2)
    def getResult(index, point):
        point = ee.Geometry.Point(point['coordinates'])
        region = point.buffer(params['buffer']).bounds()

        if params['format'] in ['png', 'jpg']:
            url = image.getThumbURL(
                {
                    'region': region,
                    'dimensions': params['dimensions'],
                    'format': params['format'],
                }
            )
        else:
            url = image.getDownloadURL(
                {
                    'region': region,
                    'dimensions': params['dimensions'],
                    'format': params['format'],
                }
            )

        if params['format'] == "GEO_TIFF":
            ext = 'tif'
        else:
            ext = params['format']

        r = requests.get(url, stream=True)
        if r.status_code != 200:
            r.raise_for_status()

        out_dir = os.path.abspath(params['out_dir'])
        basename = f'_{round(point["coordinates"][0],4)}_{round(point["coordinates"][1],4)}'
        filename = f"{out_dir}/{params['prefix']}{basename}.{ext}"
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        return out_dir
        #print("Done: ", basename)


    Parallel(n_jobs=40, prefer="threads")(delayed(getResult)(i, points_gee[i]) for i in tqdm(range(0, len(points_gee))))
    #for i in tqdm(range(0, len(points_gee))):
    #    getResult(i, points_gee[i])
    
    out_dir = os.path.abspath(params['out_dir'])
    print ("All NAIP image chips downloaded")
    
    return out_dir




## Simple function to add specifically named folders into an identified folder

def change_dir(DataFolder):
    os.chdir(DataFolder)
    dirList = ['DEM', 'NAIP', 'StreamVectors', 'POINTS', 'OUTPUT']
    for x in dirList:
        if not os.path.isdir(x):
            os.mkdir(x)

            



## This function devides up a raster (large DEM in this case) into tiles.
## The num variable is used to identify the number of tiles per side,
## so if num=4, then there will be 16 total tiles.  

            
def MakeTiles(DEM_path, num):
    #get names of folder and DEM
    DEM_name = os.path.splitext(DEM_path)[0].split('/')[-1]

    
    DEM = rasterio.open(DEM_path)

    
    # Set initial state
    left_bound = DEM.bounds[0] 
    upper = DEM.bounds[3]
    
    # width and length, calculate new width and length
    width = DEM.bounds[0] - DEM.bounds[2]
    length = DEM.bounds[1] - DEM.bounds[3]
    new_width = abs(width / num)
    new_length = abs(length / num)
    
    #simple counter to keep track of the nubmer of times we've iterated.  
    counter_x = 0
    
    # will increase at num**2 speed
    name_counter = 1
    
    # set dictionary to capture all the bounding boxes
    crop_polygons_bbox = {}
    
    while counter_x < num:
        # will reset for each row
        left = left_bound
        
        for n in range(num):
    
            right = left + new_width
            lower = upper - new_length
    
            box = [[left, upper], [right, upper], [right, lower], [left, lower]]
    
            crop_polygons_bbox[name_counter] = box
    
            left += new_width
    
            name_counter += 1
    
        upper -= new_length
        counter_x += 1
        
    # make a list to catch each data frame
    geoSeries_list = []
    
    for eachBBOX in crop_polygons_bbox:
        
        # define the geometry
        part_polygon = Polygon(crop_polygons_bbox[eachBBOX])
        d = {'name':[f'{eachBBOX}',], 'geometry': [part_polygon,]}
        
        # make the geodataframe
        DataFrame = gpd.GeoDataFrame(d, crs=str(DEM.crs))
        # put it in the list
        geoSeries_list.append(DataFrame)
    
    
    #combine all geodataframes into one
    all_crop_polygons = pd.concat(geoSeries_list)
    
    clipped_path_list = []
    
    for eachPolygon in all_crop_polygons.geometry:
    
    
        left = eachPolygon.bounds[0] 
        lower = eachPolygon.bounds[1]
        right = eachPolygon.bounds[2]
        upper = eachPolygon.bounds[3]
        f_name_coord = f'left_{str(round(left, 2))}_upper_{str(round(upper, 2))}_right_{str(round(right, 2))}_lower_{str(round(lower, 2))}'
    
        out_tif = f'OUTPUT/{DEM_name}_{f_name_coord}.tif'
    
        # Set NAIP bounding box
        point_box = [[left, upper], [right, upper], [right, lower], [left, lower]]
    
        geometries = [{'type': 'Polygon',
                       'coordinates': [point_box]}]
    
    
        out_img, out_transform = mask(DEM, shapes=geometries, crop=True,  all_touched=True)
    
    
        # Copy the metadata
        out_meta = DEM.meta.copy()
        epsg_code = int(DEM.crs.data['init'][5:])
        out_meta.update({"driver": "GTiff",
                         "height": out_img.shape[1],
                         "width": out_img.shape[2],
                         "transform": out_transform,
                         "crs": DEM.crs}
                       ) 
    
        clipped_path_list.append(out_tif)
        with rasterio.open(out_tif, "w", **out_meta) as dest:
    
            dest.write(out_img)
    return clipped_path_list




## This function will crop a large raster (for instance a downloaded DEM) 
## using a shapefile's bounds to speed up processsing.  



def crop_byBoundary(raster_path, cropBoundary_SHP):
    
    raster = rasterio.open(raster_path)
    raster_name = os.path.split(raster_path)[-1].split('.')[0]

    try:        
        crop_poly = gpd.read_file(cropBoundary_SHP)
        crop_project = crop_poly.to_crs("EPSG:3857")

        crop_buff = crop_project.buffer(1000, cap_style=3)
        crop_repro = crop_buff.to_crs(raster.crs)
        

        out_tif = os.path.join((os.path.split(raster_path)[0]),f'{raster_name}_cropped.tif')

        left = crop_repro.total_bounds[0] 
        lower = crop_repro.total_bounds[1]
        right = crop_repro.total_bounds[2]
        upper = crop_repro.total_bounds[3]

        # Set raster bounding box
        raster_box = [[left, upper], [right, upper], [right, lower], [left, lower], [left, upper]]

        geometries = [{'type': 'Polygon',
                           'coordinates': [raster_box]}]


        out_img, out_transform = mask(raster, shapes=geometries, crop=True,  all_touched=True)


        # Copy the metadata
        out_meta = raster.meta.copy()
        epsg_code = int(raster.crs.data['init'][5:])
        out_meta.update({"driver": "GTiff",
                         "height": out_img.shape[1],
                         "width": out_img.shape[2],
                         "transform": out_transform,
                         "crs": raster.crs}
                       ) 


        with rasterio.open(out_tif, "w", **out_meta) as dest:
            dest.write(out_img)

        return out_tif
    except: 
        return
    


## This function removes older versions of the DEMs downloaded from the 
## National Map. It requires dates to be included in the file name
## and deletes the older names.  


def deleteOldFiles(folder):
    dirFolder = os.path.join((os.getcwd()),folder)


    root_name = []
    for eachFile in glob.glob(dirFolder + '/*'):
        a = '_'.join(os.path.split(eachFile)[-1].split('.')[0].split('_')[:3])
        if a not in root_name:
            root_name.append(a)


    newestFile = []
    for name in root_name:
        x = max(glob.glob(os.path.join(dirFolder,name+'*')))
        newestFile.append(x)

    for file in glob.glob(os.path.join(dirFolder,'*')):
        if file not in newestFile:
            os.remove(file)
    
    return glob.glob(os.path.join(dirFolder,'*'))





#=============================================================================#

## This function isn't used.  It's a more complicated version of above.
## It might be more useful with very large datasets and memory management.  

def makeExtent_fromSHP(shp_path):

    shp = gpd.read_file(shp_path)
    shp_name = os.path.split(InputData)[-1].split('.')[0]
    
    left = shp.total_bounds[0] 
    lower = shp.total_bounds[1]
    right = shp.total_bounds[2]
    upper = shp.total_bounds[3]

    # Set raster bounding box
    shp_box = [[left, upper], [right, upper], [right, lower], [left, lower]]

    # define the geometry
    shp_poly = Polygon(shp_box)
    d = {'name':[f'{shp_name}',], 'geometry': [shp_poly,]}

    # make the geodataframe
    shp_extent = gpd.GeoDataFrame(d, crs=str(shp.crs))
    
    return shp_extent

#=============================================================================#



