"""Data structures for different storm cell representations."""

from shapely.geometry import MultiPolygon, Polygon


class StormCell:
    """Defines a storm cell represented by a (multi)polygon."""

    def __init__(self, obstime, identifier, polygon, srid):
        """Initialize a storm cell.

        Parameters
        ----------
        obstime : datetime.datetime
            The observation time of the cell.
        identifer : object
            The identifier of the cell.
        polygon : shapely.geometry.MultiPolygon or shapely.geometry.Polygon
            The polygon defining the cell.
        srid : int
            The spatial reference identifier (SRID) of the storm cell geometry.
        """
        self.__obstime = obstime
        self.__identifier = identifier
        if isinstance(polygon, Polygon):
            self.__polygon = polygon.copy()
        else:
            self.__polygon = MultiPolygon(polygon)
        self.__attrs = {}
        self.__srid = srid

    def __getitem__(self, key):
        """Get the given attribute. Raise KeyError if the attribute does not exist."""
        return self.__attrs[key]

    def __setitem__(self, key, value):
        """Set an attribute to the given value."""
        self.__attrs[key] = value

    @property
    def area(self):
        """The area of the cell."""
        return self.__polygon.area

    @property
    def attributes(self):
        """The attributes of the cell."""
        return self.__attrs.keys()

    @property
    def centroid(self):
        """The centroid of the cell."""
        return self.__polygon.centroid

    @property
    def identifier(self):
        """The identifier of the cell."""
        return self.__identifier

    @property
    def obstime(self):
        """The observation time of the cell."""
        return self.__obstime

    @property
    def polygon(self):
        """The polygon defining the cell."""
        return self.__polygon

    @property
    def srid(self):
        """The spatial reference identifier (SRID) of the storm cell geometry."""
        return self.__srid
