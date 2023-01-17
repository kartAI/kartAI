class Region:
    def __init__(self, poly):
        self.poly = poly
        self.minx, self.maxx, self.miny, self.maxy = poly.GetEnvelope()

