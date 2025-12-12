#!/usr/bin/env python3
"""
Script to prepare workshop data files (COG and FlatGeobuf) from existing sources.

This script requires:
- WMS API key for aerial imagery
- PostgreSQL credentials for building data
- GDAL/OGR with COG driver support

Usage:
    python prepare_workshop_data.py --area sand --output-dir ./workshop_data
    python prepare_workshop_data.py --area all --output-dir ./workshop_data
"""

import argparse
import os
import sys
from pathlib import Path

# Workshop areas with bounding boxes (EPSG:25832)
WORKSHOP_AREAS = {
    "sand": {
        "name": "Sand (Jessheim)",
        "bbox": (618296.0, 6668145.0, 620895.0, 6670133.0),
        "description": "Sand area outside of Jessheim"
    },
    "skoyen": {
        "name": "Skøyen (Oslo)",
        "bbox": (593150.9, 6643812.3, 596528.0, 6644452.2),
        "description": "Skøyen area in Oslo"
    },
    "midtbyen": {
        "name": "Midtbyen (Trondheim)",
        "bbox": (568372.6, 7033216.7, 570820.4, 7034223.7),
        "description": "Midtbyen area in Trondheim"
    }
}

def create_cog_from_wms(area_key, bbox, output_path, wms_api_key):
    """
    Create a Cloud Optimized GeoTIFF from WMS source.
    
    Args:
        area_key: Key identifying the area
        bbox: Tuple of (minx, miny, maxx, maxy) in EPSG:25832
        output_path: Path where COG file should be saved
        wms_api_key: API key for WMS service
    """
    try:
        from osgeo import gdal
    except ImportError:
        print("ERROR: GDAL Python bindings not found. Install with: pip install gdal")
        sys.exit(1)
    
    minx, miny, maxx, maxy = bbox
    width = int((maxx - minx) / 0.25)  # 0.25m resolution
    height = int((maxy - miny) / 0.25)
    
    # WMS URL
    wms_url = (
        f"WMS:https://waapi.webatlas.no/wms-orto/"
        f"?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS=ortofoto&STYLES=new_up"
        f"&CRS=EPSG:25832"
        f"&BBOX={minx},{miny},{maxx},{maxy}"
        f"&WIDTH={width}&HEIGHT={height}"
        f"&FORMAT=image/tiff"
        f"&api_key={wms_api_key}"
    )
    
    print(f"Downloading aerial imagery for {area_key}...")
    print(f"  Bbox: {bbox}")
    print(f"  Size: {width}x{height} pixels")
    print(f"  Output: {output_path}")
    
    # Create COG with compression
    ds = gdal.Translate(
        str(output_path),
        wms_url,
        format='COG',
        creationOptions=[
            'COMPRESS=JPEG',
            'QUALITY=85',
            'BLOCKSIZE=512',
            'OVERVIEW_RESAMPLING=AVERAGE'
        ]
    )
    
    if ds is None:
        print(f"ERROR: Failed to create COG for {area_key}")
        return False
    
    ds = None  # Close dataset
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Created COG: {output_path} ({file_size_mb:.1f} MB)")
    return True


def create_flatgeobuf_from_postgis(area_key, bbox, output_path, db_config):
    """
    Create a FlatGeobuf file from PostgreSQL/PostGIS source.
    
    Args:
        area_key: Key identifying the area
        bbox: Tuple of (minx, miny, maxx, maxy) in EPSG:25832
        output_path: Path where FlatGeobuf file should be saved
        db_config: Dict with PostgreSQL connection info
    """
    try:
        from osgeo import ogr
    except ImportError:
        print("ERROR: OGR Python bindings not found. Install with: pip install gdal")
        sys.exit(1)
    
    minx, miny, maxx, maxy = bbox
    
    # PostgreSQL connection string
    pg_connection = (
        f"PG:host={db_config['host']} "
        f"port={db_config['port']} "
        f"dbname={db_config['database']} "
        f"user={db_config['user']} "
        f"password={db_config['password']}"
    )
    
    # SQL query to get buildings in bbox
    sql_query = (
        f"SELECT geom, osm_id, building "
        f"FROM {db_config['table']} "
        f"WHERE ST_Intersects(geom, "
        f"ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 25832))"
    )
    
    print(f"Extracting building data for {area_key}...")
    print(f"  Bbox: {bbox}")
    print(f"  Output: {output_path}")
    
    # Open source
    src_ds = ogr.Open(pg_connection)
    if src_ds is None:
        print(f"ERROR: Failed to connect to PostgreSQL database")
        return False
    
    src_layer = src_ds.ExecuteSQL(sql_query)
    if src_layer is None:
        print(f"ERROR: Failed to execute SQL query")
        return False
    
    feature_count = src_layer.GetFeatureCount()
    print(f"  Features: {feature_count}")
    
    # Create FlatGeobuf
    driver = ogr.GetDriverByName('FlatGeobuf')
    dst_ds = driver.CreateDataSource(str(output_path))
    
    # Copy layer
    dst_layer = dst_ds.CopyLayer(src_layer, 'buildings')
    
    # Cleanup
    src_ds.ReleaseResultSet(src_layer)
    src_ds = None
    dst_ds = None
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Created FlatGeobuf: {output_path} ({file_size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare workshop data files (COG and FlatGeobuf)"
    )
    parser.add_argument(
        '--area',
        choices=list(WORKSHOP_AREAS.keys()) + ['all'],
        required=True,
        help='Workshop area to prepare (or "all" for all areas)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./workshop_data'),
        help='Directory to save output files'
    )
    parser.add_argument(
        '--wms-api-key',
        help='WMS API key (or set NK_WMS_API_KEY env var)'
    )
    parser.add_argument(
        '--db-password',
        help='PostgreSQL password (or set OSM_DB_PWD env var)'
    )
    
    args = parser.parse_args()
    
    # Get credentials
    wms_api_key = args.wms_api_key or os.environ.get('NK_WMS_API_KEY')
    db_password = args.db_password or os.environ.get('OSM_DB_PWD')
    
    if not wms_api_key:
        print("ERROR: WMS API key required (use --wms-api-key or set NK_WMS_API_KEY)")
        sys.exit(1)
    
    if not db_password:
        print("ERROR: Database password required (use --db-password or set OSM_DB_PWD)")
        sys.exit(1)
    
    # Database config
    db_config = {
        'host': 'postgresql-dev-kartai.postgres.database.azure.com',
        'port': '5432',
        'database': 'kartai_opendata',
        'user': 'kartai_opendata_ro@postgresql-dev-kartai',
        'password': db_password,
        'table': 'public.osm_buildings'
    }
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which areas to process
    areas_to_process = (
        WORKSHOP_AREAS.keys() if args.area == 'all' 
        else [args.area]
    )
    
    print(f"\n{'='*60}")
    print("Workshop Data Preparation")
    print(f"{'='*60}\n")
    
    success_count = 0
    fail_count = 0
    
    for area_key in areas_to_process:
        area_info = WORKSHOP_AREAS[area_key]
        print(f"\nProcessing: {area_info['name']}")
        print(f"Description: {area_info['description']}\n")
        
        # File paths
        cog_path = args.output_dir / f"{area_key}_ortofoto.tif"
        fgb_path = args.output_dir / f"{area_key}_buildings.fgb"
        
        # Create COG
        if create_cog_from_wms(area_key, area_info['bbox'], cog_path, wms_api_key):
            success_count += 1
        else:
            fail_count += 1
            continue
        
        # Create FlatGeobuf
        if create_flatgeobuf_from_postgis(area_key, area_info['bbox'], fgb_path, db_config):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count} files created, {fail_count} failed")
    print(f"Output directory: {args.output_dir.absolute()}")
    print(f"{'='*60}\n")
    
    # Print next steps
    print("Next steps:")
    print("1. Upload files to hosting service (Azure Blob, S3, etc.)")
    print("2. Update workshop config with file URLs")
    print("3. Test workshop notebook with new config")
    print()


if __name__ == '__main__':
    main()
