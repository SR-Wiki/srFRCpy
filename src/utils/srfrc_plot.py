"""Backward-compatible wrappers for consolidated srFRC map rendering."""
from src.utils.plotting import ResolutionPlotter


def display_maximum(analysis, index):
    """Return a common maximum for background comparisons."""
    return ResolutionPlotter.srfrc_display_max(analysis, index)


def plot_srfrc_results(analysis, show=True):
    """Render selected-frame srFRC maps without recalculating data."""
    return ResolutionPlotter.srfrc_maps(analysis, show=show)
