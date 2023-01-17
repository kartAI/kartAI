from osgeo import gdal, ogr, osr
import imageio
import os
import requests
import sys
import time
import shutil
import numpy as np
import math
import scipy as sp
import env

ogr.UseExceptions()
gdal.UseExceptions()


class TileGrid:
    """Specification of the geometry of a tiled image grid"""

    def __init__(self, srid, x0, y0, dx, dy=None):
        """Create an image tile grid in a coordinate system given by srid,
        using x0, y0 as origin and each image tile having sidelength dx, dy.
        Select an origin south-west of your data area so that all tile indices will be positive"""
        self.srid = srid
        self.srs = osr.SpatialReference()
        self.srs.ImportFromEPSG(srid)
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy if dy is not None else dx

    def image_geom(self, i, j):
        """Return the extent if an image tile with grid index i, j as
        minx, miny, maxx, maxy"""
        minx = self.x0 + i * self.dx
        miny = self.y0 + j * self.dy
        maxx = minx + self.dx
        maxy = miny + self.dy
        return minx, miny, maxx, maxy

    def generate_ij(self, region):
        """Generator function for iterating through all i, j indexes for image
        tiles intersecting the specified area.
        Area is currently a rectangle specified by a list or tuple
        [minx, miny, maxx, maxy], but more complex shapes may be allowed in the future"""
        mini = int(math.floor((region.minx - self.x0) / self.dx))
        minj = int(math.floor((region.miny - self.y0) / self.dy))
        maxi = int(math.ceil((region.maxx - self.x0) / self.dx))
        maxj = int(math.ceil((region.maxy - self.y0) / self.dy))

        # Create ring, fill with zero
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(0, 0)
        ring.AddPoint(0, 0)
        ring.AddPoint(0, 0)
        ring.AddPoint(0, 0)
        ring.AddPoint(0, 0)

        # Create polygon
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometryDirectly(ring)
        poly.AssignSpatialReference(region.poly.GetSpatialReference())

        j = minj
        while j <= maxj:
            i = mini
            while i <= maxi:
                minx, miny, maxx, maxy = self.image_geom(i, j)
                ring.SetPoint_2D(0, minx, miny)
                ring.SetPoint_2D(1, maxx, miny)
                ring.SetPoint_2D(2, maxx, maxy)
                ring.SetPoint_2D(3, minx, maxy)
                ring.SetPoint_2D(4, minx, miny)

                if region.poly.Intersects(poly):
                    yield i, j
                i += 1
            j += 1


class Tile:
    """
    Container for image data, initialized with a file path and / or an image raster and geometry.
    When initialized from file path the image data are loaded lazily, when needed
    """

    def __init__(self, image_source, i, j, tile_size):
        self._image_source = image_source
        self._i = i
        self._j = j
        self._tile_size = tile_size
        self._array = None
        self._srs_wkt = None
        self._geo_transform = None

    @property
    def file_path(self):
        return self._image_source.image_path(self._i, self._j, self._tile_size)

    @property
    def array(self):
        """Return the image array, load if necessary"""
        self._load()
        return self._array

    @property
    def srs_wkt(self):
        """Return the image spatial reference system, load if necessary"""
        self._load()
        return self._srs_wkt

    @property
    def geo_transform(self):
        """Return the image geographic transform, load if necessary"""
        self._load()
        return self._geo_transform

    def to_json(self):
        return {
            "image_source": self._image_source.source_name,
            "i": self._i,
            "j": self._j,
            "tile_size": self._tile_size
        }

    @classmethod
    def from_json(cls, jn, sources):
        return cls(sources[jn["image_source"]], jn["i"], jn["j"], jn["tile_size"])

    @staticmethod
    def tileset_to_json(tileset):
        jn = {}
        image_sources = set()
        json_examples = []
        for item in tileset:
            conv_item = {}
            for k, v in item.items():
                conv_item[k] = v.to_json()
                image_sources.add(v._image_source)
            json_examples.append(conv_item)
        jn["Examples"] = json_examples

        cache_root = None
        tile_grid = None
        json_im_srs = []
        for im_source in image_sources:
            if cache_root is None:
                cache_root = im_source.cache_root
            if tile_grid is None:
                tile_grid = im_source.tile_grid
            json_im_srs += im_source.to_json()
        jn["ImageSources"] = json_im_srs

        jn["CacheRoot"] = cache_root
        jn["TileGrid"] = {
            "srid": tile_grid.srid,
            "x0": tile_grid.x0,
            "y0": tile_grid.y0,
            "dx": tile_grid.dx,
            "dy": tile_grid.dy
        }

        return jn

    @classmethod
    def tileset_from_json(cls, jn):
        tileset = []
        image_sources = {}
        tile_grid = TileGrid(jn["TileGrid"]["srid"],
                        jn["TileGrid"]["x0"], jn["TileGrid"]["y0"],
                        jn["TileGrid"]["dx"], jn["TileGrid"]["dy"])

        for source_config in jn["ImageSources"]:
            source = ImageSourceFactory.create(
                jn["CacheRoot"], tile_grid, image_sources, source_config)
            if source:
                image_sources[source_config["name"]] = source
            else:
                print("Unknown source:", source_config, file=sys.stderr)

        for item in jn["Examples"]:
            conv_item = {}
            for k, v in item.items():
                conv_item[k] = cls.from_json(v, image_sources)
            tileset.append(conv_item)

        return tileset

    def _load(self):
        if self._array is not None and self._srs_wkt and self._geo_transform:
            return
        file_path = self.file_path
        if not file_path:
            raise ValueError("No file_path?")
        if self._image_source.cache_root is not None and os.path.exists(file_path):
            data_source = gdal.Open(file_path)
            self._geo_transform = data_source.GetGeoTransform()
            self._srs_wkt = data_source.GetSpatialRef().ExportToWkt()
            self._array = data_source.ReadAsArray()
        else:
            # *img_geom, = self._image_source.tile_grid.image_geom(self._i, self._j)
            self._array, self._srs_wkt, self._geo_transform = \
                self._image_source.load_tile(self._i, self._j, self._tile_size)
            # self._image_source.load_image(file_path, *img_geom, self._tile_size)
            if self._image_source.cache_root is not None and not os.path.exists(file_path):
                self._save()

    def _save(self):
        if not (self._array is not None and self._srs_wkt and self._geo_transform):
            raise ValueError("No data?")

        file_path = self.file_path
        if not file_path:
            raise ValueError("No file_path?")

        gdal_type = None
        if self._array.dtype == np.single:
            gdal_type = gdal.GDT_Float32
        elif self._array.dtype == np.double:
            self._array = np.array(self._array, dtype=np.single)
            gdal_type = gdal.GDT_Float32
        elif self._array.dtype == np.byte or self._array.dtype == np.ubyte:
            gdal_type = gdal.GDT_Byte

        driver = None
        gdal_options = []
        if file_path.endswith(".tif") or file_path.endswith(".tiff"):
            driver = gdal.GetDriverByName("GTiff")
            gdal_options = ['COMPRESS=LZW', 'PREDICTOR=2']

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data_source = driver.Create(file_path,
                                    self._array.shape[0], self._array.shape[1],
                                    self._array.shape[2] if len(
                                        self._array.shape) > 2 else 1,
                                    gdal_type, gdal_options)

        data_source.SetGeoTransform(self._geo_transform)
        data_source.SetProjection(self._srs_wkt)
        data_source.GetRasterBand(1).WriteArray(self._array)


class ImageSourceFactory:
    """Factory for image sources"""
    @staticmethod
    def create(cache_root, tile_grid, image_sources, layer_spec):
        type = layer_spec["type"]

        if type == "WMSImageSource":
            return WMSImageSource(cache_root, tile_grid, layer_spec)
        elif type == "ImageFileImageSource":
            return ImageFileImageSource(cache_root, tile_grid, layer_spec)
        elif type == "WCSImageSource":
            return WCSImageSource(cache_root, tile_grid, layer_spec)
        elif type == "PostgresImageSource":
            return PostgresImageSource(cache_root, tile_grid, layer_spec)
        elif type == "VectorFileImageSource":
            return VectorFileImageSource(cache_root, tile_grid, layer_spec)
        elif type == "CompositeImageSource":
            return CompositeImageSource(cache_root, tile_grid, layer_spec, image_sources)
        return None


class ImageSource:
    """
    Abstract base class for image sources.
    Create a cacheable image source from a directory path to a cache_root,
    a tile_grid describing the tile grid geometry,
    data for an image_source where images are fetced (typically, a WMS or a database)
    """

    format_ext = {
        "image/tiff": "tif",
        "GeoTiff": "tif",
        "image/jpeg": "jpg"
    }

    def __init__(self, cache_root, tile_grid, layer_spec):
        self.cache_root = cache_root
        self.tile_grid = tile_grid
        self.source_name = layer_spec["name"]
        self.img_format = layer_spec["image_format"]
        self.layer_spec = layer_spec

    def image_path(self, i, j, tile_size):
        """Return the file path to a cached image with tile grid index i, j.

        An example of a file path:
        'cache_root/image_source_name/25832_450000_6200000_100_100/512/32_74.tif'
        Where cache_root is the root directory specified in the ImageeSet constructor,
        image_source_name is the name specified in the constructor of the ImageSource,
        25832 is the epsg code of the spatial reference system specified in the construcor of the tile grid.
        450000 is the x coordinate of the tile grid origin,
        6200000 is the y coordinate of the tile grid origin,
        100 and 100 are the x and y extent of a tile,
        512 is the image size (in pixels) and
        32 and 74 is the tile grid indices i, j in the x and y direction"""

        retval = os.path.join(self.source_name,
                              f'{self.tile_grid.srid}_{self.tile_grid.x0}_{self.tile_grid.y0}_{self.tile_grid.dx}_{self.tile_grid.dy}',
                              f'{tile_size}',
                              f'{i}_{j}.{self.format_ext[self.img_format]}')

        if self.cache_root is not None:
            retval = os.path.join(self.cache_root, retval)
        return retval

    def load_tile(self, i, j, tile_size):
        img_path = self.image_path(i, j, tile_size)
        *img_geom, = self.tile_grid.image_geom(i, j)
        return self.load_image(img_path, *img_geom, tile_size)

    def load_image(self, image_path, minx, miny, maxx, maxy, tile_size):
        """Load or generate an image with the geometry given by
        minx, miny, maxx, maxy, srid, tile_size from the image source and store into the image_path"""
        raise NotImplementedError(
            f"Class {self.__class__.__name__}.load_image is not implemented")

    def to_json(self):
        return [self.layer_spec]


class WMSImageSource(ImageSource):
    """Fetch images from an WMS service"""

    def __init__(self, cache_root, tile_grid, layer_spec):
        super().__init__(cache_root, tile_grid, layer_spec)

        self.base_url = layer_spec["url"]
        self.layers = layer_spec["layers"]
        self.styles = layer_spec["styles"] if "styles" in layer_spec else []
        self.api_key = layer_spec["api_key"] if "api_key" in layer_spec else None

    def load_image(self, image_path, minx, miny, maxx, maxy, tile_size):

        params = {
            'api_key': env.get_env_variable(self.api_key) if self.api_key else None,
            'request': 'GetMap',
            'layers': ",".join(self.layers),
            'styles': ",".join(self.styles) if self.styles else None,
            'width': tile_size,
            'height': tile_size,
            'srs': f'epsg:{self.tile_grid.srid}',
            'format': self.img_format,
            'bbox': f'{minx}, {miny}, {maxx}, {maxy}'
        }

        # Do request
        req = requests.get(self.base_url, stream=True, params=params,
                           headers=None, timeout=None)

        # Handle response
        if not req:
            print("Something went very wrong in WMSImageSource", file=sys.stderr)
        elif req.status_code == 200:
            print("request status is 200")
            if req.headers['content-type'] == self.img_format:
                # If response is OK and an image, save image file
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                with open(image_path, 'wb') as out_file:
                    shutil.copyfileobj(req.raw, out_file)

                data_source = gdal.Open(image_path)
                return data_source.GetGeoTransform(), data_source.GetSpatialRef().ExportToWkt(), data_source.ReadAsArray()

            else:
                # If no image, print error to stdout
                print("Content-type: ", req.headers['content-type'],
                      " url: ", req.url, " Content: ", req.text, file=sys.stderr)

        # Use existing
        elif req.status_code == 304:
            data_source = gdal.Open(image_path)
            return data_source.GetGeoTransform(), data_source.GetSpatialRef().ExportToWkt(), data_source.ReadAsArray()

        # Handle error
        else:
            print("Status: ", req.status_code,
                  " url: ", req.url, file=sys.stderr)

        return None, None, None


class WCSImageSource(ImageSource):
    """Fetch images from a WCS service"""

    def __init__(self, cache_root, tile_grid, layer_spec):
        super().__init__(cache_root, tile_grid, layer_spec)
        self.base_url = layer_spec["url"]
        self.img_srs = {}
        self.coverage = layer_spec['coverage']

    def load_image(self, image_path, minx, miny, maxx, maxy, tile_size):
        params = {
            'service': 'wcs',
            'version': '1.0.0',
            'request': 'getCoverage',
            'coverage': self.coverage,
            'format': self.img_format,
            'width': tile_size,
            'height': tile_size,
            'crs': f'EPSG:{self.tile_grid.srid}',
            'bbox': f'{minx}, {miny}, {maxx}, {maxy}'
        }

        # Do request
        req = None
        wait = 1
        for i in range(10):
            try:
                req = requests.get(self.base_url, stream=True, params=params,
                                   headers=None, timeout=None)
            except:
                pass
            if not req:
                time.sleep(wait)
                wait += 1
            else:
                if req.status_code == 200:
                    break
                else:
                    time.sleep(wait)
                    wait += 1

        # Handle response
        if not req:
            print("Something went very wrong in WCSImageSource", file=sys.stderr)
            return None, None, None
        elif req.status_code == 200:
            print("request status is 200")

        data = imageio.imread(req.content, ".tif")

        # If response is OK and an image, save image file
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        target_ds = gdal.GetDriverByName('GTiff').Create(
            str(image_path),
            tile_size, tile_size, 1, gdal.GDT_Float32,
            ['COMPRESS=LZW', 'PREDICTOR=2'])

        geo_transform = (
            minx, (maxx - minx) / tile_size, 0,
            maxy, 0, (miny - maxy) / tile_size,
        )
        target_ds.SetGeoTransform(geo_transform)

        srs_wkt = self.tile_grid.srs.ExportToWkt()
        target_ds.SetProjection(srs_wkt)
        target_ds.GetRasterBand(1).WriteArray(data)
        return data, srs_wkt, geo_transform


class OGRImageSource(ImageSource):
    """
    "Burn" vectordata into images. Uses OGR. Usually used as a base class for Postgres or vector file sources
    """

    def __init__(self, cache_root, tile_grid, layer_spec):
        super().__init__(cache_root, tile_grid, layer_spec)

        # The layer
        self.layer = None

        # Setup attribute filter
        self.attribute_filter_json = layer_spec["attribute_filter"] if "attribute_filter" in layer_spec else None
        if self.attribute_filter_json:
            if isinstance(self.attribute_filter_json, (list, tuple)):
                self.attribute_filter = [af for af in self.attribute_filter_json]
            else:
                self.attribute_filter = [self.attribute_filter_json]
        else:
            self.attribute_filter = [None]

    def open(self):
        raise NotImplementedError(
            f"Class {self.__class__.__name__}.open() is not implemented")

    def load_image(self, image_path, minx, miny, maxx, maxy, tile_size):
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        target_ds = gdal.GetDriverByName('GTiff').Create(
            image_path,
            tile_size, tile_size, 1, gdal.GDT_Byte, ['COMPRESS=LZW', 'PREDICTOR=2'])
        geo_transform = (
            minx, (maxx - minx) / tile_size, 0,
            maxy, 0, (miny - maxy) / tile_size,
        )
        target_ds.SetGeoTransform(geo_transform)
        srs_wkt = self.tile_grid.srs.ExportToWkt()
        target_ds.SetProjection(srs_wkt)

        if self.layer is None:
            self.open()

        # Create ring
        if self.layer.GetSpatialRef():
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(minx, miny)
            ring.AddPoint(maxx, miny)
            ring.AddPoint(maxx, maxy)
            ring.AddPoint(minx, maxy)
            ring.AddPoint(minx, miny)

            # Create polygon
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            poly.AssignSpatialReference(self.tile_grid.srs)
            poly.TransformTo(self.layer.GetSpatialRef())

            self.layer.SetSpatialFilter(poly)

        feature_count = self.layer.GetFeatureCount()
        print('feature count: ', feature_count)
        # Fill raster - Label saved when GDAL completes rasterization
        if isinstance(self.attribute_filter, (list, tuple)):
            for ix, attr_f in enumerate(self.attribute_filter):
                bv = 1 + ix
                if isinstance(attr_f, (list, tuple)):
                    bv = attr_f[0]
                    attr_f = attr_f[1]
                self.layer.SetAttributeFilter(attr_f)
                if gdal.RasterizeLayer(target_ds, [1], self.layer, burn_values=[bv], options=['ALL_TOUCHED=TRUE']) != 0:
                    err = "lar"
                    raise Exception("error rasterizing layer: %s" % err)
        elif isinstance(self.attribute_filter, str):
            self.layer.SetAttributeFilter(self.attribute_filter)
            if gdal.RasterizeLayer(target_ds, [1], self.layer, burn_values=[1], options=['ALL_TOUCHED=TRUE']) != 0:
                err = "lar"
                raise Exception("error rasterizing layer: %s" % err)
        else:
            if gdal.RasterizeLayer(target_ds, [1], self.layer, burn_values=[1], options=['ALL_TOUCHED=TRUE']) != 0:
                err = "lar"
                raise Exception("error rasterizing layer: %s" % err)

        arr = target_ds.ReadAsArray()
        target_ds = None  # Save to file
        return arr, srs_wkt, geo_transform


class PostgresImageSource(OGRImageSource):
    """A Postgres / PostGIS specialization of the OGRImageSource. Has database connection convenience initializer"""

    def __init__(self, cache_root, tile_grid, layer_spec):
        super().__init__(cache_root, tile_grid, layer_spec)

    def open(self):
        # Connect to database
        # print('connecting to db')
        connectionPwd = env.get_env_variable(self.layer_spec['passwd'])
        connString = f"PG: host={self.layer_spec['host']} port={self.layer_spec['port']} dbname={self.layer_spec['database']} " + \
                     f"user={self.layer_spec['user']} password={connectionPwd}"
        self.data_source = ogr.Open(connString)
        # print('created connection', self.conn)

        if "sql" in self.layer_spec:
            self.layer = self.data_source.ExecuteSQL(self.layer_spec["sql"])
        elif "table" in self.layer_spec:
            self.layer = self.data_source.GetLayerByName(self.layer_spec["table"])
        if not self.layer:
            raise ValueError("No layer?")


class VectorFileImageSource(OGRImageSource):
    """A vector database specialization of the OGRImageSource. Has database connection convenience initializer"""

    def __init__(self, cache_root, tile_grid, layer_spec):
        super().__init__(cache_root, tile_grid, layer_spec)

    def open(self):
        self.data_source = ogr.Open(self.layer_spec["file_path"])
        if "layer" in self.layer_spec:
            self.layer = self.data_source.GetLayerByName(self.layer_spec["layer"])
        else:
            self.layer = self.data_source.GetLayerByIndex()


class ImageFileImageSource(ImageSource):
    """
    Get image tiles from a geo-referenced image.

    This should work for all gdal-supported formats, including VRT (for tiled image mosaics)
    """

    def __init__(self, cache_root, tile_grid, layer_spec):
        super().__init__(cache_root, tile_grid, layer_spec)
        self.file_path = layer_spec["file_path"]
        self.data_source = gdal.Open(self.file_path)
        self.img_srs = {}

    def load_image(self, image_path, minx, miny, maxx, maxy, tile_size):
        """Load or generate an image with the geometry given by
        minx, miny, maxx, maxy, srid, tile_size from the image source and store into the image_path"""
        target_ds = gdal.GetDriverByName('GTiff').Create(
            str(image_path),
            tile_size, tile_size, self.data_source.RasterCount, gdal.GDT_Byte,
            ['COMPRESS=LZW', 'PREDICTOR=2'])
        geo_transform = (
            minx, (maxx - minx) / tile_size, 0,
            maxy, 0, (miny - maxy) / tile_size,
        )
        target_ds.SetGeoTransform(geo_transform)

        srs_wkt = self.data_source.GetSpatialRef().ExportToWkt()
        target_ds.SetProjection(srs_wkt)
        gdal.ReprojectImage(self.data_source, target_ds,
                            self.data_source.GetSpatialRef().ExportToWkt(), srs_wkt, gdal.GRA_Bilinear)

        return target_ds.ReadAsArray(), srs_wkt, geo_transform

class Composition:
    """Create an image from operations on other images"""

    def __init__(self, content, scale=None):
        self.content = content
        self.scale = scale

    def get_tile(self, i, j, tile_size):
        raise NotImplementedError(
            f"Class {self.__class__.__name__}.get_tile is not implemented")

    @staticmethod
    def create(config, image_sources):
        """Composition Factory"""
        scale = config["scale"] if "scale" in config else None
        if "source" in config:
            return SourceComposition(image_sources[config["source"]], scale)
        elif "and" in config:
            return AndComposition([Composition.create(c, image_sources) for c in config["and"]], scale)
        elif "or" in config:
            return OrComposition([Composition.create(c, image_sources) for c in config["or"]], scale)
        elif "add" in config:
            return AddComposition([Composition.create(c, image_sources) for c in config["add"]], scale)
        elif "convolution" in config:
            return ConvolveComposition(Composition.create(config["convolution"], image_sources),
                                       config["convolution"]["radius"] if "radius" in config["convolution"] else None)
        elif "clip" in config:
            return ClipComposition(Composition.create(config["clip"], image_sources),
                                   config["clip"]["min"] if "min" in config["clip"] else None,
                                   config["clip"]["max"] if "max" in config["clip"] else None)
        else:
            raise ValueError("Unknown composition", config)


class SourceComposition(Composition):
    """"Get data from a source, pass on to next operation (possibly scaled)"""

    def __init__(self, content, scale=None):
        super(SourceComposition, self).__init__(content, scale)

    def get_tile(self, i, j, tile_size):
        arr, srs_wkt, geo_transform = self.content.load_tile(i, j, tile_size)
        if self.scale is not None:
            arr = arr * self.scale
        return arr, srs_wkt, geo_transform


class ConvolveComposition(Composition):
    """"Get data from a source, pass on to next operation (possibly scaled)"""

    def __init__(self, content, radius=None):
        super(ConvolveComposition, self).__init__(content)
        self.radius = radius

        iradius = int(math.ceil(radius))
        self.kernel = np.zeros((2*iradius+1, 2*iradius+1), dtype=np.single)
        for i in range(2*iradius+1):
            for j in range(2*iradius+1):
                if math.hypot(i - radius, j - radius) < radius:
                    self.kernel[i, j] = 1

    def get_tile(self, i, j, tile_size):
        arr, srs_wkt, geo_transform = self.content.get_tile(i, j, tile_size)
        if self.radius is not None:
            arr = sp.signal.convolve2d(arr, self.kernel,  mode="same")

        if self.scale is not None:
            arr = arr * self.scale

        return arr, srs_wkt, geo_transform


class ClipComposition(Composition):
    def __init__(self, content, min_clip=None, max_clip=None):
        super(ClipComposition, self).__init__(content)
        self.min = -np.inf if min_clip is None else min_clip
        self.max = np.inf if max_clip is None else max_clip

    def get_tile(self, i, j, tile_size):
        arr, srs_wkt, geo_transform = self.content.get_tile(i, j, tile_size)
        arr = np.clip(arr, self.min, self.max)
        return arr, srs_wkt, geo_transform


class AndComposition(Composition):
    """Probability "and" between images with [0, 1] values"""

    def __init__(self, content, scale=None):
        super(AndComposition, self).__init__(content, scale)

    def get_tile(self, i, j, tile_size):
        content_it = iter(self.content)
        cnt = next(content_it)
        arr, srs_wkt, geo_transform = cnt.get_tile(i, j, tile_size)
        arr = np.array(arr)
        for cnt in content_it:
            next_img = cnt.get_tile(i, j, tile_size)
            arr *= next_img.array

        if self.scale is not None:
            arr *= self.scale

        return arr, srs_wkt, geo_transform


class OrComposition(Composition):
    """Probability "or" between images with [0, 1] values"""

    def __init__(self, content, scale=None):
        super(OrComposition, self).__init__(content, scale)

    def get_tile(self, i, j, tile_size):
        content_it = iter(self.content)
        cnt = next(content_it)
        arr, srs_wkt, geo_transform = cnt.get_tile(i, j, tile_size)
        arr = np.array(arr)
        for cnt in content_it:
            next_arr, next_wkt, next_geo_trans = cnt.get_tile(i, j, tile_size)
            arr = arr + next_arr - arr * next_arr

        if self.scale is not None:
            arr *= self.scale

        return arr, srs_wkt, geo_transform


class AddComposition(Composition):
    """Add containing images"""

    def __init__(self, content, scale=None):
        super(AddComposition, self).__init__(content, scale)

    def get_tile(self, i, j, tile_size):
        content_it = iter(self.content)
        cnt = next(content_it)
        if isinstance(cnt,  Composition):
            arr, srs_wkt, geo_transform = cnt.get_tile(i, j, tile_size)
            arr = np.array(arr)
        else:
            arr, srs_wkt, geo_transform = cnt[1].get_tile(i, j, tile_size)
            arr = cnt[0] * arr

        for cnt in content_it:
            if isinstance(cnt, Composition):
                next_img = cnt.get_tile(i, j, tile_size)
                arr += next_img.array
            else:
                next_img = cnt[1].get_tile(i, j, tile_size)
                arr += cnt[0] * next_img.array

        if self.scale is not None:
            arr *= self.scale

        return arr, srs_wkt, geo_transform


class CompositeImageSource(ImageSource):
    """Create a composition of images from various sources"""

    def __init__(self, cache_root, tile_grid, layer_spec, image_sources):
        super().__init__(cache_root, tile_grid, layer_spec)

        self.composition = Composition.create(layer_spec["composition"], image_sources)

    def load_tile(self, i, j, tile_size):
        """Find an image with tile grid indices i, j and return the cached image path.
              Load the image if necessary. After this method is called you are assured that the
              image is loaded at the given file path address"""

        if self.cache_root is None:
            arr, srs_wkt, geo_transform =  self.composition.get_tile(i, j, tile_size)
        else:
            img_path = self.image_path(i, j, tile_size)
            if os.path.exists(img_path):
                data_source = gdal.Open(img_path)
                geo_transform = data_source.GetGeoTransform()
                srs_wkt = data_source.GetSpatialRef().ExportToWkt()
                arr = data_source.ReadAsArray()
            else:
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                arr, srs_wkt, geo_transform = self.composition.get_tile(i, j, tile_size)
        return arr, srs_wkt, geo_transform

    def _find_dependent_sources(self, cmp):
        retval = set()
        if isinstance(cmp, SourceComposition):
            retval.add(cmp.content)
        elif isinstance(cmp, Composition):
            retval |= self._find_dependent_sources(cmp.content)
        elif hasattr(cmp, '__iter__'):
            for ch in cmp:
                retval |= self._find_dependent_sources(ch)
        return retval

    def to_json(self):
        retval = super().to_json()
        sub = self._find_dependent_sources(self.composition)
        for s in sub:
            retval = s.to_json() + retval

        return retval