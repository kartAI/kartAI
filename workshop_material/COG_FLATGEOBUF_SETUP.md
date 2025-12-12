# Using COGs and FlatGeobuf for Workshop Data

## Overview

This document describes how to set up the workshop to use file-based data sources (Cloud Optimized GeoTIFFs and FlatGeobuf files) instead of WMS and PostgreSQL database.

## Benefits

- **No API keys required**: Eliminates need for WMS API key
- **No database credentials**: No PostgreSQL access needed
- **Offline capable**: Workshop can run with cached files
- **Simpler setup**: Fewer dependencies and credentials to manage
- **Better performance**: Direct file access can be faster than web services
- **Reproducible**: Workshop data is versioned and immutable

## Data Requirements

### 1. Aerial Imagery (COG)

**Format**: Cloud Optimized GeoTIFF (COG)
- COGs are regular GeoTIFFs with internal tiling and overviews
- Enable efficient streaming and partial reading over HTTP
- Can be created from any GeoTIFF using `gdal_translate` with COG driver

**Example areas for workshop**:
- Sand/Jessheim: bbox (618296.0, 6668145.0, 620895.0, 6670133.0) in EPSG:25832
- Skøyen/Oslo: bbox (593150.9, 6643812.3, 596528.0, 6644452.2) in EPSG:25832
- Midtbyen/Trondheim: bbox (568372.6, 7033216.7, 570820.4, 7034223.7) in EPSG:25832

**Creating COGs**:
```bash
# From existing GeoTIFF
gdal_translate input.tif output_cog.tif \
  -of COG \
  -co COMPRESS=JPEG \
  -co QUALITY=85 \
  -co BLOCKSIZE=512

# From WMS (download for workshop area)
gdal_translate \
  "WMS:https://waapi.webatlas.no/wms-orto/?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ortofoto&CRS=EPSG:25832&BBOX=618296,6668145,620895,6670133&WIDTH=2048&HEIGHT=2048&FORMAT=image/tiff" \
  workshop_area_sand.tif -of COG
```

### 2. Building Labels (FlatGeobuf)

**Format**: FlatGeobuf (.fgb)
- Efficient cloud-native vector format
- Supports spatial indexing
- Smaller file size than GeoJSON or Shapefile
- Can be streamed over HTTP with spatial filtering

**Creating FlatGeobuf**:
```bash
# From PostgreSQL database
ogr2ogr -f FlatGeobuf workshop_buildings.fgb \
  PG:"host=postgresql-dev-kartai.postgres.database.azure.com \
      dbname=kartai_opendata \
      user=kartai_opendata_ro \
      password=xxx" \
  -sql "SELECT geom FROM public.osm_buildings WHERE ST_Intersects(geom, ST_MakeEnvelope(618296, 6668145, 620895, 6670133, 25832))"

# From Shapefile
ogr2ogr -f FlatGeobuf buildings.fgb buildings.shp

# From GeoJSON
ogr2ogr -f FlatGeobuf buildings.fgb buildings.geojson
```

## Hosting Options

### Option 1: Azure Blob Storage (Recommended)
- Upload COG and FlatGeobuf files to Azure Blob Storage
- Make container public or use SAS tokens
- GDAL/OGR can read directly from `/vsiaz/` paths or HTTPS URLs

```python
# Example file paths
image_file = "https://storageaccount.blob.core.windows.net/workshop-data/sand_ortofoto.tif"
label_file = "https://storageaccount.blob.core.windows.net/workshop-data/sand_buildings.fgb"
```

### Option 2: GitHub Releases
- Good for smaller datasets (< 100 MB per file)
- Free and publicly accessible
- Limited by GitHub's file size restrictions

### Option 3: AWS S3
- Similar to Azure Blob Storage
- Can use `/vsis3/` paths or HTTPS URLs
- Requires bucket to be public or use signed URLs

## Configuration

### 1. Create Workshop Config File

See `workshop_file_based_config_template.json` for the structure.

Key changes from WMS/PostgreSQL config:
```json
{
  "ImageSources": [
    {
      "name": "OrtofotoFile",
      "type": "ImageFileImageSource",
      "image_format": "image/tiff",
      "file_path": "https://storage.example.com/workshop/aerial.tif"
    },
    {
      "name": "OSMByggFile", 
      "type": "VectorFileImageSource",
      "image_format": "image/tiff",
      "file_path": "https://storage.example.com/workshop/buildings.fgb",
      "srid": 25832
    }
  ]
}
```

### 2. Update Notebook

The workshop notebook needs minimal changes:
1. Remove WMS API key and database password from secrets
2. Update config file reference to use file-based config
3. No other changes needed - kartAI code already supports file sources

```python
# OLD (WMS/PostgreSQL)
os.environ['NK_WMS_API_KEY'] = # Insert API key here
os.environ['OSM_DB_PWD'] = # Insert DB PWD

create_training_data(
    config_file_path="kartAI/config/dataset/osm_bygg.json",
    ...
)

# NEW (File-based)
# No secrets needed if files are public!

create_training_data(
    config_file_path="kartAI/config/dataset/workshop_file_based.json",
    ...
)
```

## Implementation Checklist

- [ ] Create or obtain COG files for workshop areas
- [ ] Create or obtain FlatGeobuf files for workshop areas  
- [ ] Upload files to hosting service (Azure Blob, S3, etc.)
- [ ] Create production config file with actual file URLs
- [ ] Update workshop notebook to use file-based config
- [ ] Remove WMS and PostgreSQL credential requirements
- [ ] Test complete workshop flow with file-based sources
- [ ] Update workshop README with new setup instructions

## File Size Estimates

For a typical workshop area (2.6km x 2km at 0.25m resolution):
- **Aerial imagery (COG)**: 50-200 MB (depending on compression)
- **Building polygons (FlatGeobuf)**: 1-10 MB

Total per area: ~60-210 MB
For 3 workshop areas: ~180-630 MB total

## GDAL/OGR Configuration

The kartAI code uses GDAL/OGR which supports various cloud storage options:

```python
# Azure Blob with SAS token
"/vsiaz/container/file.tif"

# HTTPS (public files)
"/vsicurl/https://storage.example.com/file.tif"

# S3
"/vsis3/bucket/file.tif"
```

For the workshop, HTTPS URLs to public files are simplest:
- No additional authentication
- Works in Google Colab without configuration
- Easy to test and debug

## Testing

Test file access before workshop:
```python
from osgeo import gdal

# Test COG access
ds = gdal.Open("https://storage.example.com/workshop/aerial.tif")
if ds:
    print(f"COG loaded: {ds.RasterXSize}x{ds.RasterYSize}")
else:
    print("Failed to load COG")

# Test FlatGeobuf access  
ds = ogr.Open("https://storage.example.com/workshop/buildings.fgb")
if ds:
    layer = ds.GetLayer(0)
    print(f"FlatGeobuf loaded: {layer.GetFeatureCount()} features")
else:
    print("Failed to load FlatGeobuf")
```

## Next Steps

1. **Decision needed**: Choose hosting solution (Azure Blob recommended)
2. **Data preparation**: Create COG and FlatGeobuf files for workshop areas
3. **Upload**: Host files with public or SAS token access
4. **Config**: Create production config with actual URLs
5. **Notebook**: Update to use file-based config
6. **Test**: Complete workshop dry-run with file sources
