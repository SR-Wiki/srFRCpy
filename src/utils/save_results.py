"""Plot and export utilities invoked by demos for optional results/ output."""
from pathlib import Path
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from src.utils.analysis_results import DRI_GUIDANCE,dri_interpretation,resolve_plot,current_result_index
from src.utils.sfsc_guidance import ANGLE_GUIDANCE,fit_range_advice
from src.utils.polar_plot import polar_plot
from src.utils.shifted_jet import sjet_colorbar
from src.utils.srfrc_plot import display_maximum


def _results_directory():
    """Create the demo results directory in the current working directory.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    pathlib.Path
        The existing or newly created results/ directory.
    """
    directory=Path.cwd()/'results'
    directory.mkdir(exist_ok=True)
    return directory


def _write_summary(directory,module,rows,metadata):
    """Save demo summary rows and parameters as CSV and JSON.
    
    Parameters
    ----------
    directory : pathlib.Path
        Demo results directory.
    module : str
        Analysis name used in the exported filenames: sFRC, sFSC, or srFRC.
    rows : list of dict
        One summary dictionary per analysed frame or pair.
    metadata : dict
        Parameters and interpretation notes to include in the JSON export.
    
    Returns
    -------
    None
        Writes <module>_results.csv and <module>_results.json to directory.
    """
    with (directory/f'{module}_results.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]) if rows else [])
        writer.writeheader();writer.writerows(rows)
    def serial(value):
        """Convert nested NumPy values into JSON-compatible Python values.
        
        Parameters
        ----------
        value : object
            Value to validate or serialize, as described by the function.
        
        Returns
        -------
        object
            Recursively converted data; non-finite floating values become None.
        """
        if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [serial(v) for v in value]
        if isinstance(value,np.ndarray):return serial(value.tolist())
        if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
        if isinstance(value,np.integer):return int(value)
        if isinstance(value,np.bool_):return bool(value)
        return value
    with (directory/f'{module}_results.json').open('w',encoding='utf-8') as file:
        json.dump(serial(dict(results=rows,metadata=metadata)),file,indent=2,allow_nan=False)


def save_sfrc_results(result,plot='auto'):
    """Export completed sFRC curves from the demo layer.
    
    Parameters
    ----------
    result : object
        Returned analysis object or ordered collection of results.
    plot : bool or str
        'auto' enables previews for the current-frame result; False hides them.
        Batch and individual runs preview only the current frame/pair. {default: 'auto'}
    
    Returns
    -------
    pathlib.Path
        Current-directory results/ folder containing CSV/JSON summaries and per-frame
        NPZ curves, plus optional PNG plots.
    
    Notes
    -----
    PNG plots show only the current frame/pair; numerical batches remain complete. No input filename or
    output-directory argument is needed.
    """
    directory=_results_directory()
    curves=result if isinstance(result,list) else [result]
    plot=resolve_plot(plot,curves[0].parameters.get('batch_processing',False))
    parameters=curves[0].parameters
    plot_index=current_result_index(len(curves),parameters.get('batch_processing',False),
        parameters.get('current_frame',0),parameters.get('single_frame',True),notify=False)
    _write_summary(directory,'sFRC',[c.summary for c in curves],curves[0].parameters)
    for result_index,curve in enumerate(curves):
        label=curve.summary['frame']
        data={k:v for k,v in vars(curve).items() if isinstance(v,np.ndarray)}
        np.savez_compressed(directory/f'sFRC_{label}_correlation-curve.npz',**data)
        if plot and result_index==plot_index:
            curve.plot_results(curve.summary['threshold'],
                save_path=directory/f'sFRC_{label}_correlation-curve.png',
                title=f'sFRC - correlation curve - {label}',show=False)
    return directory


def save_sfsc_results(result,plot_fitting_curve=False):
    """Export the sFSC model, directional samples, and interpretation guidance.
    
    Parameters
    ----------
    result : object
        Returned analysis object or ordered collection of results.
    plot_fitting_curve : bool
        Additionally export the DRI elliptical-fit PNG. {default: False}
    
    Returns
    -------
    pathlib.Path
        Current-directory results/ folder containing summary CSV/JSON, DRI-data NPZ,
        E_DRI guidance text, a polar PNG for an available result and an optional fit PNG.
    """
    directory=_results_directory()
    metadata=dict(result.parameters,E_DRI_guidance=DRI_GUIDANCE,
                  fitting_angle_guidance=ANGLE_GUIDANCE,
                  axial_resolution_note=getattr(result,'interpretation_note',''))
    _write_summary(directory,'sFSC',[result.summary],metadata)
    np.savez_compressed(directory/'sFSC_DRI-data.npz',
        angles_rad=np.asarray(result.angles if result.angles is not None else [],dtype=float),
        cutoffs_per_um=np.asarray(result.cutoffs if result.cutoffs is not None else [],dtype=float),
        lateral_resolution_nm=result.summary['lateral_resolution_nm'],
        axial_resolution_nm=result.summary['axial_resolution_nm'],E_DRI=result.goodness_of_fit)
    with (directory/'sFSC_E-DRI-guidance.txt').open('w',encoding='utf-8') as file:
        file.write(DRI_GUIDANCE+'\n'+ANGLE_GUIDANCE+'\n')
        error=result.summary['E_DRI_percent']/100
        file.write(f'E_DRI={100*error:.2f}% ({dri_interpretation(error)})\n')
        advice=fit_range_advice(error)
        if advice:file.write(advice+'\n')
        if metadata['axial_resolution_note']:file.write(metadata['axial_resolution_note']+'\n')
    if np.isfinite(result.summary['axial_resolution_nm']):
        polar_plot(result,save_path=directory/'sFSC_lateral-axial-resolution.png',show=False)
        if plot_fitting_curve:
            fig,ax=plt.subplots()
            xx=np.linspace(0,result.xy_cutoff,200)
            ax.plot(xx,result.z_cutoff*np.sqrt(np.maximum(0,1-(xx/result.xy_cutoff)**2)),color='#E67373')
            ax.scatter(result.cutoffs*np.cos(result.angles),result.cutoffs*np.sin(result.angles),s=15)
            ax.set(xlabel='Spatial frequency XY (1/um)',ylabel='Spatial frequency Z (1/um)',
                   title='sFSC - DRI elliptical fit')
            fig.savefig(directory/'sFSC_DRI-elliptical-fit.png',dpi=160,bbox_inches='tight')
            plt.close(fig)
    return directory


def save_srfrc_results(analysis,plot='auto'):
    """Export final rolling maps and their per-frame statistics from the demo layer.
    
    Parameters
    ----------
    analysis : RollingFourierRingCorrelation
        Completed rolling analysis returned by execute.
    plot : bool or str
        'auto' enables previews for the current-frame result; False hides them.
        Batch and individual runs preview only the current frame/pair. {default: 'auto'}
    
    Returns
    -------
    pathlib.Path
        Current-directory results/ folder containing summary CSV/JSON, map NPZ, float32
        TIFF stacks, and optional per-frame PNG maps.
    
    Notes
    -----
    File names include srFRC, the processing stage, frame/pair identifiers where
    applicable, and the background setting. Exported statistics were calculated on
    the cleaned native rolling maps before upsampling. TIFF/NPZ/PNG contain the
    restored maps, whose mean can differ. PNG previews use only the current frame/pair.
    """
    directory=_results_directory()
    plot=resolve_plot(plot,analysis.parameters.get('batch_processing',False))
    metadata=dict(analysis.parameters,map_axes='TYX',
        statistics='from native rolling maps before cubic-spline upsampling, after background cleanup and the 2.3x original-pixel-size lower bound',
        statistics_stage=analysis.statistics_stage,
        statistics_map_shape=list(analysis.statistics_map_shape),
        map_interpolation_backend=analysis.map_interpolation_backend,
        final_background_thresholds=analysis.final_background_thresholds)
    _write_summary(directory,'srFRC',analysis.metrics,metadata)
    np.savez_compressed(directory/'srFRC_results.npz',**analysis.rfrc_maps)
    for setting,maps in analysis.rfrc_maps.items():
        rows=[row for row in analysis.metrics if row['background_setting']==setting]
        with (directory/f'srFRC_{setting}_results.csv').open('w',newline='',encoding='utf-8-sig') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]) if rows else [])
            writer.writeheader(); writer.writerows(rows)
        tifffile.imwrite(directory/f'srFRC_{setting}_resolution-map.tif',maps.astype(np.float32),
            photometric='minisblack',metadata=dict(axes='TYX',unit='nm',pixel_size_nm=analysis.pixel_size,
                statistics_stage=analysis.statistics_stage))
        if plot:
            plot_index=current_result_index(len(maps),analysis.parameters.get('batch_processing',False),
                analysis.parameters.get('current_frame',0),analysis.parameters.get('single_frame',True),notify=False)
            for result_index,(row,data) in enumerate(zip(rows,maps)):
                if result_index!=plot_index:
                    continue
                fig,ax=plt.subplots(figsize=(8,5))
                artist=ax.imshow(data,cmap=sjet_colorbar(),vmin=0,vmax=display_maximum(analysis,result_index))
                label=row['frame']
                ax.set_title(f'srFRC - resolution map - {label}\nBackground threshold = {row.get("background_threshold", float("nan")):.1f}');ax.axis('off')
                fig.colorbar(artist,ax=ax,label='Resolution (nm)')
                fig.savefig(directory/f'srFRC_{label}_{setting}_resolution-map.png',dpi=160,bbox_inches='tight')
                plt.close(fig)
    return directory
