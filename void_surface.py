"""Analysis of cubegen CUBEs to visualise void surfaces."""

import numpy as np
# import pickle as p
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from copy import copy

__author__ = "Filip T. Szczypiński"


class Cube:
    """
    Represents a CUBE stored in a cubegen ``.cube`` file.

    Attributes
    ----------
    atoms : ndarray
        Atoms in the ``.cube`` file in the format [atomic_number, x, y, z].

    centre_of_mass : ndarray
        An array with the [x, y, z] coordinates (in Angstrom) of the centre of
        mass of the molecule stored in the ``.cube`` file.

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

    def __init__(self, path, cube_type=''):
        """
        Initialise a `Cube` from a cubegen ``.cube`` file.

        Parameters
        ----------
        path : str
            Path to the ``.cube`` file.

        cube_type : str, optional
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

    def generate_isosurface(self, isovalue=0.0004, rtol=0.01, atol=1E-30):
        """
        Generate the isosurface for a given value.

        Parameters
        ----------
        isovalue : float, optional
            The target isovalue of the points on the isosurface (the default is
            0.0004).

        rtol : float, optional
            The relative tolerance parameter (the default is 1E-2).

        atol : float, optional
            The absolute tolerance parameter (the default is 1E-30).

        Returns
        -------
        `Isosurface`
            The isovalue surface.

        Notes
        -----
        The default isovalue corresponds to 0.0004 a.u. of the total electron
        density distribution, which has been found to accurately define the
        volume of the molecule.

        """
        z_lines = self._get_z_lines()
        indices = list()
        values = list()

        with open(self.cube_path, 'r') as f:
            # Skip the comments and Cube properties.
            for _ in range(self.natoms + 6):
                next(f)

            for x in range(self.nx):
                for y in range(self.ny):
                    for z in range(z_lines):
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

            return Isosurface(self, isovalue, rtol, atol, indices, values)

    def get_value(self, x, y, z):
        """
        Get the value of the point with CUBE indices [x, y, z].

        Parameters
        ----------
        x, y, z : int
            Indices in the three CUBE directions.

        Returns
        -------
        float
            Value at the point with CUBE indices [x, y, z].

        """
        z_lines = self._get_z_lines()

        with open(self.cube_path, 'r') as f:
            # Skip the comments, properties, and atomic positions.
            skip = self.natoms + 6
            # Get to the required x value
            skip += x * self.ny * z_lines
            # Get to the required y value
            skip += y * z_lines
            # Get to the required z line
            skip += int(z/6)

            for _ in range(skip):
                next(f)

            values = f.readline().split()

        return values[z % 6]

    def _get_z_lines(self):
        """
        Get the number of lines per z index in the ``.cube`` file.

        Each values line of the ``.cube`` file contains up to six values and
        it is not filled if z indices run out.

        Returns
        -------
        int
            The number of lines per z value.

        """
        return int(self.nz / 6) if self.nz % 6 == 0 else int(self.nz / 6) + 1

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

    def get_coords(self):
        """
        Get the [x, y, z] coordinates of the points on the surface.

        Returns
        -------
        ndarray
            The array containing the [x, y, z] coordinates (in Angstrom) of the
            points on the surface.

        """
        if self.indices.size == 0:
            return self.indices
        else:
            coords = self.indices * self.parent_cube.unit + \
                self.parent_cube.origin
            return coords

    def get_void_surface(self, removeHs=False):
        """
        Approximate which surface points lie within the internal void.

        Parameters
        ----------
        removeHs : bool, optional
            If True, hydrogen atoms will be removed prior to the approximation
            of the void points (default is False).

        Returns
        -------
        list of [ndarray, float]
            An array of the void surface points in the form [(x, y, z), value],
            where the x, y, z are the **CUBE indices** (not the coordinates).

        list of [ndarray, float]
            An array of the outside surface points in the form
            [(x, y, z), value], where the x, y, z are the **CUBE indices**
            (not the coordinates).

        Notes
        -----
        Currently, the method uses KDTrees to find the five atoms nearest to
        the surface point in question and then determines whether any of those
        atoms lies closer to the molecule's centre of mass than the point. If
        all atoms are further away from the centre than the point, then the
        point is deemed to be inside the cavity. Albeit crude, this protocol
        seems reasonably accurate at estimating the surface points within the
        void of non-convex cages.

        Furthermore, the method copies the original Surface instance in order
        to keep the remaining properties the same, hence can be thught of as
        returning a "subset" of the original surface under investigation.

        """
        if removeHs:
            atoms = self.parent_cube.atoms[self.parent_cube.atoms[:, 0] > 1]
        else:
            atoms = self.parent_cube.atoms

        coords = self.get_coords()

        # Find five nearest atoms to each surface point.
        kdtree = KDTree(atoms[:, 1:])
        distances, atom_ids = kdtree.query(coords, 5)

        # Calculate distances from CoM to each atom.
        a_dist = dict()
        for a, a_xyz in enumerate(atoms[:, 1:]):
            a_dist[a] = euclidean(self.parent_cube.centre_of_mass, a_xyz)

        # Compare distances of the atoms and surface points to CoM.
        void_points = list()
        void_values = list()
        outside_points = list()
        outside_values = list()

        for n, point in enumerate(coords):
            near = atom_ids[n]
            pdist = euclidean(self.parent_cube.centre_of_mass, point)

            inside = False if any(a_dist[i] < pdist for i in near) else True

            if inside:
                void_points.append(self.indices[n])
                void_values.append(self.values[n])

            else:
                outside_points.append(self.indices[n])
                outside_values.append(self.values[n])

        void = copy(self)
        void.indices = np.array(void_points)
        void.values = np.array(void_values)

        outside = copy(self)
        outside.indices = np.array(outside_points)
        outside.values = np.array(outside_values)

        return void, outside

    def map_surface(self, mapped_cube):
        """
        Map values from a ``.cube`` file onto the surface.

        Parameters
        ----------
        mapped_cube : Cube
            The Cube holding the values to be mapped onto the surface.

        Returns
        -------
        `MappedSurface`
            A surface with new values mapped onto the existing surface.

        """
        map_values = list()
        z_lines = mapped_cube._get_z_lines()

        with open(mapped_cube.cube_path, 'r') as f:
            for _ in range(mapped_cube.natoms + 6):
                next(f)

            zs = list(map(float, f.readline().split()))
            line = 0

            for x, y, z in self.indices:
                target = x * mapped_cube.ny * z_lines + y * z_lines + int(z/6)
                while line < target:
                    zs = list(map(float, f.readline().split()))
                    line += 1
                map_values.append(zs[z % 6])

        return MappedSurface(self, mapped_cube, map_values)


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
        self.atol = atol
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
        self.values = np.array(values)
