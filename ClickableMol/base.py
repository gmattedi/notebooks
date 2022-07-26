from io import BytesIO
from typing import Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from matplotlib.figure import Figure
from rdkit import Chem
from rdkit.Chem import Draw

Draw.rdDepictor.SetPreferCoordGen(True)


class ClickableMol:
    """
    Display an interactive matplotlib figure of an RDKit molecule.
    Click on atoms to select or deselect them.

    The set returned by `.plot` contains the indices of the atoms that
    are currently selected
    """

    def plot(
            self, mol: Chem.Mol, tolerance: int = 25, width: int = 200, height: int = 200,
            figsize: Tuple[int, int] = (3, 3), **marker_kwargs
    ) -> Tuple[Figure, Set[int]]:
        """
        Display the molecule

        Args:
            mol (Chem.Mol)
            tolerance (int): Radius for registering a click to an atom, in pixels
            width (int)
            height (int)
            figsize (Tuple[int, int]): matplotlib figure size
            **marker_kwargs: kwargs for atom markers, passed to `plt.scatter`

        Returns:
            fig (Figure)
            clicked (Set[int]): Set of clicked atom indices, updated in real time

        """
        img, coords = self.get_drawing(mol, width=width, height=height)
        fig, clicked = self._render(img, coords=coords, figsize=figsize, tolerance=tolerance, **marker_kwargs)
        return fig, clicked

    @staticmethod
    def get_drawing(mol: Chem.Mol, width: int = 200, height: int = 200) -> \
            Tuple[PngImageFile, np.ndarray]:
        """
        Get the PIL image drawing of a molecule, and the 2D positions of atoms in the drawing

        Args:
            mol (Chem.Mol)
            width (int)
            height (int)

        Returns:
            img (PngImageFile): PIL image
            coords (np.ndarray): Nx2 array of 2D atom coordinates in the drawing
        """

        d = Draw.rdMolDraw2D.MolDraw2DCairo(width, height)
        d.DrawMolecule(mol)
        d.FinishDrawing()

        handle = BytesIO()
        handle.write(d.GetDrawingText())

        img = Image.open(handle)

        coords = np.array([
            [d.GetDrawCoords(i).x, d.GetDrawCoords(i).y]
            for i in range(mol.GetNumAtoms())
        ])

        # handle.close()

        return img, coords

    @staticmethod
    def _render(img: PngImageFile, coords: np.ndarray, figsize: Tuple[int, int] = (3, 3), tolerance: int = 25,
                **marker_kwargs) -> Tuple[Figure, Set[int]]:
        """
        Render the interactive plot

        Args:
            img (PngImageFile): PIL image from `.get_drawing`
            coords (np.ndarray): 2D coordinates from `.get_drawing`
            figsize (Tuple[int, int]): matplotlib figure size
            tolerance (int): Radius for registering a click to an atom, in pixels
            **marker_kwargs: kwargs for atom markers, passed to `plt.scatter

        Returns:
            fig (Figure): matplotlib figure
            clicked_set (Set[int]): Set of clicked atom indices, updated in real time

        """
        # Marker defaults
        marker_kwargs['s'] = marker_kwargs.get('s', 200)
        marker_kwargs['alpha'] = marker_kwargs.get('alpha', 0.5)
        marker_kwargs['color'] = marker_kwargs.get('color', 'orange')

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        ax.imshow(img)

        clicked_set: Set[int] = set()

        markers = plt.scatter([], [], **marker_kwargs)

        def onclick(event):
            click = (event.xdata, event.ydata)

            dist = np.linalg.norm(coords - click, axis=1)
            clicked = dist <= tolerance
            if clicked.sum() > 0:
                clicked_idx = int(np.where(clicked)[0][0])
                if clicked_idx not in clicked_set:
                    clicked_set.update([clicked_idx])
                else:
                    clicked_set.discard(clicked_idx)

                markers.set_offsets(coords[list(clicked_set)])

        fig.canvas.mpl_connect('button_press_event', onclick)
        return fig, clicked_set
