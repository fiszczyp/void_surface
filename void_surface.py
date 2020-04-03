"""Analysis of cubegen CUBE files to visualise void surfaces."""

import numpy as np
# import pickle as p
# from scipy.spatial import KDTree
# from scipy.spatial.distance import euclidean

__author__ = "Filip T. Szczypiński"


class Cube:
    """
    Represents a CUBE file.

    Attributes
    ----------
    atoms : list
        Atoms saved in the cube file in the format [atomic_number, x, y, z].

    cube_path : str
        Path to the CUBE file containing the data.

    cube_type : str
        Type of the calculation stored in the CUBE file.

    natoms : int
        Number of atoms in the CUBE file.

    nx, ny, nz : int
        Number of points in the three CUBE directions.

    origin : ndarray
        Initial point of the CUBE file in the format [x0, y0, z0].

    unit : ndarry
        Unit vector within the CUBE in the format [x1, y1, z1].

    """

    def __init__(self, path, cube_type):
        """
        Initialise a :class:`Cube` from a cubegen ``.cube`` file.

        Parameters
        ----------
        path : :class:`str`
            Path to the CUBE file.

        cube_type : :class:`str`
            Type of the CUBE file (e.g. 'potential' or 'density').

        """
        self.cpath = path
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
            self.n_z = int(u[0])
            z1 = np.array([float(x) for x in u[1:]])

            self.unit = x1 + y1 + z1

            # Read atoms.
            self.atoms = list()
            for _ in range(self.natoms):
                line = f.readline().split()
                anum, _, x, y, z = int(line[0]), *map(float, line[1:])
                self.atoms.append([anum, x, y, z])

    def set_path(self, path):
        """
        Update the path to the ``.cube`` file.

        Allows the user to update the path to the ``.cube`` file. Might be
        useful if the Cube instance was loaded (e.g. unpickled) and the
        original ``.cube`` file was moved to another location.

        Parameters
        ----------
        path : str
            New path to the ``.cube`` file.

        """
        self.cube_path = path


class Surface:
    """
    Surface descriptors.

    Attributes
    ----------
    parent_cube : :class:`Cube`
        The CUBE from which the surface is generated.

    """

    def __init__(self, parent_cube, indices, values):
        """
        Initialise a class:`Surface`.

        Parameters
        ----------
        parent_cube : Cube
            The CUBE from which the surface is generated.

        indices : iterable of tuples of int
            The x, y, z indices of the points on the surface.

        values : iterable of float
            The values of the points on the surface.

        """
        self.parent_cube = parent_cube
        self.indices = indices
        self.values = values