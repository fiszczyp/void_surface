"""Analysis of cubegen CUBEs to visualise void surfaces."""

import numpy as np
# import pickle as p
# from scipy.spatial import KDTree
# from scipy.spatial.distance import euclidean

__author__ = "Filip T. Szczypiński"


class Cube:
    """
    Represents a CUBE stored in a cubegen ``.cube`` file.

    Attributes
    ----------
    atoms : ndarray
        Atoms in the ``.cube`` file in the format [atomic_number, x, y, z].

    centre_of_mass : ndarray
            An array with the [x, y, z] coordinates of the centre of mass.

    cube_path : str
        Path to the ``.cube`` file containing the data.

    cube_type : str
        Type of the calculation stored in the ``.cube`` file.

    natoms : int
        Number of atoms in the ``.cube`` file.

    nx, ny, nz : int
        Number of points in the three CUBE directions.

    origin : ndarray
        Initial point of the CUBE in the format [x0, y0, z0].

    unit : ndarry
        Unit vector within the CUBE in the format [x1, y1, z1].

    """

    def __init__(self, path, cube_type):
        """
        Initialise a `Cube` from a cubegen ``.cube`` file.

        Parameters
        ----------
        path : str
            Path to the ``.cube`` file.

        cube_type : str
            Type of the ``.cube`` file (e.g. 'potential' or 'density').

        """
        self.cube_path = path
        self.cube_type = cube_type

        with open(f'{path}', 'r') as f:
            # The first two lines are comments.
            for _ in range(2):
                next(f)

            # Number of atoms and origin of the CUBE.
            u = f.readline().split()
            self.natoms = int(u[0])
            self.origin = np.array([float(x) for x in u[1:4]])

            # x increments and unit vector.
            u = f.readline().split()
            self.nx = int(u[0])
            x1 = np.array([float(x) for x in u[1:]])

            # y increments and unit vector.
            u = f.readline().split()
            self.ny = int(u[0])
            y1 = np.array([float(x) for x in u[1:]])

            # z increments and unit vector.
            u = f.readline().split()
            self.nz = int(u[0])
            z1 = np.array([float(x) for x in u[1:]])

            self.unit = x1 + y1 + z1

            # Read atoms.
            atoms = list()
            for _ in range(self.natoms):
                line = f.readline().split()
                anum, _, x, y, z = int(line[0]), *map(float, line[1:])
                atoms.append([anum, x, y, z])

            self.atoms = np.array(atoms)
            self.centre_of_mass = np.average(self.atoms[:, 1:], axis=0)

    def generate_isosurface(self, isovalue, rtol=0.01, atol=1E-30):
        """
        Generate the isosurface for a given value.

        Parameters
        ----------
        isovalue : float
            The target isovalue of the points on the isosurface.

        rtol : float, optional
            The relative tolerance parameter (the default is 1E-2).

        atol : float, optional
            The absolute tolerance parameter (the default is 1E-30).

        Returns
        -------
        `Isosurface`
            The isovalue surface.

        """
        # There are maximum 6 z values per line
        lines = int(self.nz / 6) if self.nz % 6 == 0 else int(self.nz / 6) + 1
        indices = list()
        values = list()

        with open(self.cube_path, 'r') as f:
            # Skip the comments and Cube properties.
            for _ in range(self.natoms + 6):
                next(f)

            for x in range(self.nx):
                for y in range(self.ny):
                    for z in range(lines):
                        zs = map(float, f.readline().split())
                        for i, value in enumerate(zs):
                            if np.isclose(
                                value,
                                isovalue,
                                rtol=rtol,
                                atol=atol
                            ):
                                indices.append((x, y, z*6 + i))
                                values.append(value)

            return Isosurface(self, isovalue, rtol, indices, values)

    def get_centre_of_mass(self):
        """
        Get the geometric centre of mass of the molecule in the `Cube`.

        Returns
        -------
        ndarray
            An array with the [x, y, z] coordinates of the centre of mass.

        """
        return np.average(self.atoms[:, 1:], axis=0)

    def set_path(self, path):
        """
        Update the path to the ``.cube`` file.

        Allows the user to update the path to the ``.cube`` file. Might be
        useful if the `Cube` instance was loaded (e.g. unpickled) and the
        original ``.cube`` file was moved to another location.

        Parameters
        ----------
        path : str
            New path to the ``.cube`` file.

        Returns
        -------
        None

        """
        self.cube_path = path


class Surface:
    """
    Describes a surface generated from a cubegen ``.cube`` file.

    Surfaces reference the parent `Cube` instances from which they were
    generated to allow the user to quickly inspect the properties of the
    calculation, such as its type or grid spacing.

    Attributes
    ----------
    indices : ndarray
        The indices of the points on the surface in the format [x, y, z].

    parent_cube : `Cube`
        The Cube from which the surface is generated.

    values : ndarray
        The values of the points on the surface.


    Notes
    -----
    The points are defined by the [x, y, z] indices within the CUBE (**not**
    the coordinates) and hence are physically meaningless without the Cube.
    This means that Surface objects generated on `Cube`s with different grid
    spacings can only be compared once the indices are multiplied by the
    unit vectors of the correspondinge `Cube`s.

    """

    def __init__(self, parent_cube, indices, values):
        """
        Initialise a `Surface`.

        Parameters
        ----------
        parent_cube : `Cube`
            The Cube from which the surface is generated.

        indices : iterable of tuples of int
            The x, y, z indices of the points on the surface.

        values : iterable of float
            The values of the points on the surface.

        """
        self.parent_cube = parent_cube
        self.indices = np.array(indices)
        self.values = np.array(values)


class Isosurface (Surface):
    """
    Describes an isosurface of any property.

    Isosurfaces contain points of the same value (`isovalue`) of a given
    property (e.g. electronic density). Relative tolerance (`rtol`) is applied
    to account for discontinuity and numerical errors in a form
    `rtol` * `abs(isovalue)`.

    Attributes
    ----------
    atol : float
        The absolute tolerance parameter in the isovalue values.

    isovalue : float
        The isovalue for which the isosurface was generated.

    rtol : float
        The relative tolerance parameter in the isovalue values.

    """

    def __init__(self, parent_cube, isovalue, rtol, atol, indices, values):
        """
        Initialise an Isosurface.

        Parameters
        ----------
        parent_cube : `Cube`
            The Cube from which the surface is generated.

        isovalue : float
            The isovalue for which the isosurface was generated.

        rtol : float
            The relative toleramce in the isovalue value.

        atol : float
            The absolute toleramce in the isovalue value.

        indices : iterable of tuples of int
            The x, y, z indices of the points on the surface.

        values : iterable of float
            The values of the points on the surface.

        """
        self.isovalue = isovalue
        self.rtol = rtol
        super().__init__(parent_cube, indices, values)


class MappedSurface (Surface):
    """
    Describes values mapped onto another surface.

    Describes values of some property mapped onto another surface. The most
    obvious application is mapping the electrostatic potential onto an
    electornic density isosurfac to create the corresponding molecular
    electorstatic potential surface (MEPS).

    Attributes
    ----------
    surface : `Surface`
        The Surface onto which the property is mapped.

    mapped_cube : `Cube`
        The Cube containing the values mapped onto the Surface.

    """

    def __init__(self, surface, mapped_cube, values):
        """
        Initialise a `MappedSurface`.

        Parameters
        ----------
        surface : `Surface`
            The Surface onto which the property is mapped.

        mapped_cube : `Cube`
            The Cube containing the values mapped onto the Surface.

        values : iterable of float
            The values of the points on the surface.

        """
        self.surface = surface
        self.mapped_cube = mapped_cube
        self.parent_cube = self.surface.parent_cube
        self.indices = self.surface.indices
        self.values = values
