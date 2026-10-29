"""Current-frame srFRC background comparisons using the sjet palette."""
import matplotlib.pyplot as plt
import numpy as np
from src.utils.shifted_jet import sjet_colorbar


def display_maximum(analysis, index):
    """Return one common positive display maximum for the background comparisons.

    Parameters
    ----------
    analysis : RollingFourierRingCorrelation
        Completed analyser retaining all background groups.
    index : int
        Frame/pair index within each computed group's stack.

    Returns
    -------
    float
        Maximum finite positive map value, or 1 for empty foreground.
    """
    maxima=[]
    for maps in analysis.rfrc_maps.values():
        valid=maps[index][np.isfinite(maps[index]) & (maps[index]>0)]
        if valid.size:
            maxima.append(float(np.max(valid)))
    return max(maxima) if maxima else 1.0


def plot_srfrc_results(analysis, show=True):
    """Plot one row of srFRC maps with absolute background-threshold labels.

    Monitor mode shows three maps side by side, with a common sJet color
    scale [0, max_resolution]. No reference images or pixel masks are drawn.
    Non-monitor mode displays one map with its own right-hand colorbar.
    """
    index=analysis.parameters.get('plot_index')
    if index is None:
        return None
    groups=list(analysis.rfrc_maps.items())
    upper=display_maximum(analysis,index)
    is_monitor=bool(getattr(analysis,'monitor_change',False)) and len(groups)==3
    fig,axes=plt.subplots(1,len(groups),figsize=(5*len(groups),4),
                          squeeze=False,constrained_layout=True)
    for column,(setting,maps) in enumerate(groups):
        matches=[row for row in analysis.metrics if row['background_setting']==setting]
        metric=matches[index]
        label=metric['frame']
        value=float(metric.get('background_threshold',
                          getattr(analysis,'final_background_thresholds',{}).get(setting,[float('nan')])[index]))
        ax_map=axes[0,column]
        artist=ax_map.imshow(maps[index],cmap=sjet_colorbar(),vmin=0,vmax=upper,origin='upper')
        ax_map.set_title(f'srFRC - resolution map - {label}\nBackground threshold = {value:.1f}')
        ax_map.set_axis_off()
    fig.colorbar(artist,ax=list(axes[0]),fraction=0.025,pad=0.025,label='Resolution (nm)')
    if is_monitor:
        fig.suptitle('srFRC background monitoring')
    if show:
        plt.show()
    return fig
