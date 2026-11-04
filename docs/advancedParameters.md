# Advanced parameters and result interpretation — Python

Return to the [README](../README.md).

This guide applies to this Python repository. The entry points are `sFRC`,
`sFSC`, and `RollingFourierRingCorrelation` in `src/`. They accept image arrays and return results in memory. The three demos are
**Jupyter notebooks** (`demos/sFRC_demo.ipynb`, `sFSC_demo.ipynb`, and
`srFRC_demo.ipynb`); plotting/export helpers live in `src/utils/`.
The numerical functions do not write files. Open the notebooks in JupyterLab
or VS Code; do not run `.ipynb` files with `python filename.ipynb`.

Go directly to the analysis you are using:

- [sFRC — global 2D resolution](#sfrc)
- [sFSC — lateral/axial 3D resolution and DRI](#sfsc)
- [srFRC — local resolution maps](#srfrc)
- [Output files and demo utilities](#demo-outputs)

Use the original acquisition calibration. Input sampling, reported resolutions, and srFRC map values are all in **nm**.

Provide the actual acquisition spacing through `pixel_size`; it is not inferred
from the TIFF viewer's image properties. sFRC/sFSC use `None` as a placeholder,
**not** a usable spacing default. For a 2D stack use `(T,Y,X)` by default; for
sFSC use `(Z,Y,X)` with voxel spacing `[z,y,x]`. All Python spacings and
reported resolutions are in **nm**.

## How to interpret the results

| Output | Interpretation |
| --- | --- |
| Resolution (nm) | **Smaller** values indicate finer estimated resolvable detail; compare results obtained with appropriate sampling and the same analysis settings. |
| `NaN` / unavailable | No usable resolution estimate was obtained; check warnings, pixel/voxel spacing and the correlation curves instead of treating it as zero. |
| srFRC map value `0` | Background or an invalid local estimate, **not** a measured resolution of 0 nm. Zero pixels are excluded from foreground resolution statistics. |
| `E_DRI` (sFSC) | Agreement between directional cutoffs and the fitted DRI model. **Not** an error relative to ground truth or a confidence interval. |

For srFRC, background **intensity thresholds** (0–255 scale) are separate from
FRC/FSC **correlation thresholds** (`1/7 hard threshold` and `3-sigma`).
Numerical map values are in nm; colour rendering is for inspection and should
not replace quantitative TIFF/MAT/NPZ data.


| batch_processing | Computation | Display |
| --- | --- | --- |
| `True` (default) | All frames or adjacent pairs | First frame/pair |
| `False` | Only the first frame or its adjacent pair | The first frame/pair |

In non-batch mode, single-frame analysis displays the first frame; paired mode displays the first two frames. `plot=False` hides figures. Text
'true'/'false' is also accepted. With separately supplied paired arrays, non-batch mode uses the first frame from each array. sFSC treats its input as a 3D
volume and always displays an available resolution result. sFSC has no public
`plot` argument or demo `plot_results` switch; `plot_fitting_curve` controls
only the extra DRI-fit figure.

The save utilities write numerical data and summary rows for **all computed
frames**; optional PNG previews follow the one-result rule. Monitoring retains
the background comparisons for this same frame/pair. In non-batch paired mode, the first two input frames form the adjacent pair. Batch mode previews the first available pair.



## sFRC

Global resolution estimation for 2D images and stacks of 2D frames.

### sFRC input and calibration

Provide one `(Y,X)` image or one stack, normally `(T,Y,X)`.
`stack_axis=2` accepts a `(Y,X,T)` stack. Paired mode normally uses
adjacent overlapping frames from this stack; a separate matching
second image/stack remains an optional paired-input form.

### sFRC parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `pixel_size` | Required | Original calibrated XY spacing in nm: scalar or `[y,x]` |
| `single_frame` | `True` | Generate a diagonal pair from each frame; False uses adjacent frames |
| `threshold` | `'1/7 hard threshold'` | Only alternative: `'3-sigma'` |
| `mean_filter_width` | `3` | Correlation smoothing radius in frequency bins |
| `batch_processing` | `True` | Process every frame or adjacent pair |
| `stack_axis` | `0` | Frame axis: 0 for `(T,Y,X)`, 2 for `(Y,X,T)` |
| `plot` | `True` | Preview only the first frame/pair in either batch mode; False suppresses automatic plots |
| `motion_correction` | `False` | Align paired frames before correlation |

### sFRC curve extrapolation

Resolution normally uses a measured crossing of the correlation curve and
the selected threshold. When no usable crossing is available, automatic
curve extrapolation is attempted for `threshold='fixed_threshold'`.
The fitted crossing lies beyond the measured frequency range and adds no
measured information. Extrapolation is not available for `3-sigma`.

Curve extrapolation is automatic, not a public parameter.
A successful attempt prints a console notice; failed or unsupported
attempts use `RuntimeWarning`. The selected threshold is never silently
changed. An unavailable estimate remains `NaN`.

`extrapolation_triggered` records an attempt, including a failed attempt.
Inspect the returned curve and notices to determine whether it succeeded.

### sFRC stack batching and plotting

sFRC uses one stack by default. With `batch_processing=True`, an N-frame
single-frame stack produces N results; `single_frame=False` produces N-1
overlapping adjacent pairs: `(1,2)`, `(2,3)`, ..., `(N-1,N)`. Paired mode
needs at least two frames if a separate second input is omitted.

With `batch_processing=False`, only the **first** frame is analyzed.
Paired mode also uses the following frame. A separate matching second
image/stack is retained as an optional paired-input form.

`plot=True` (default) shows only the first frame/pair
regardless of batch mode; batch output still contains all computed results. Paired mode shows the first-and-second-frame calculation.
`plot=False` hides previews; numerical batches and exported data remain complete.
The demos keep numerical output separate from user-generated comparison plots and optional file exports.

### sFRC results and saved files

The compact summary contains frame/pair label, singleframe, threshold,
extrapolation_triggered, and resolution_nm. Labels use `frame0001` or
`pair0001-0002`.

| Content | Filename examples |
| --- | --- |
| Summary / metadata | `sFRC_results.csv`, `sFRC_results.json` |
| Numerical curves | `sFRC_frame0001_correlation-curve.npz` |
| Optional preview | `sFRC_frame0001_correlation-curve.png` |

## sFSC

Lateral and axial resolution estimation for 3D volumes, with DRI.

### sFSC input and calibration

Input is a `(Z,Y,X)` volume. In single-frame mode, `sFSC` generates
the working pair internally. `single_frame=False` requires a matching
second volume. Voxel spacings use `[z,y,x]` in nm.
DRI angle parameters are in degrees and measured from the XY plane.

### sFSC parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `pixel_size` | Required | Original voxel spacings `[z,y,x]` in nm |
| `z_sampling` | `False` | Manually enable complementary odd/even Z-plane sampling |
| `low_angle` | `15` | Lower DRI angle; recommended >10 degrees |
| `high_angle` | `60` | Upper DRI angle; recommended <80 degrees |
| `num_cones` | `5` | Positive integer directional-cone count; at least 5 is recommended |
| `single_frame` | `True` | Generate an XY-resampled pair from one volume |
| `threshold` | `'1/7 hard threshold'` | Only alternative: `'3-sigma'` |
| `mean_filter_width` | `3` | Correlation smoothing radius in frequency bins |
| `plot_fitting_curve` | `False` | Additionally display the DRI elliptical-fit figure; the resolution profile is always displayed |
| `xy_mode` | `'slice'` | Optional: choose `'3D'` for low-SNR data |
| `motion_correction` | `False` | Align paired inputs; not used for single-frame pair generation |

### sFSC curve extrapolation

Resolution uses a crossing between the measured correlation curve and its
threshold. Extrapolation fits a curve beyond the measured frequency range
when no usable crossing is available; it does not add measured information.

Successful extrapolation prints a stage-specific notice. Failures and
unsupported thresholds use `RuntimeWarning`; the selected threshold is never
silently changed. Internal measured-only controls support the first DRI pass
and rolling FRC and are not public switches.


### E_DRI: what the value means

DRI fits an elliptical frequency model using the lateral anchor and selected
directional cutoffs. E_DRI is the cosine-weighted relative RMS fitting error:

$$
E_{\mathrm{DRI}}=
\sqrt{\frac{\sum_i\cos\theta_i
[(k_i-k_{\mathrm{fit},i})/k_{\mathrm{fit},i}]^2}
{\sum_i\cos\theta_i}}.
$$

Here, `theta` is measured from the XY plane, `k_i` is a directional cutoff
used in the fit, and `k_fit,i` is predicted by the fitted ellipse. This checks
model consistency; it is not a ground-truth resolution error or confidence
interval. Few usable directions or an incorrect calibration can still make a
low-error result unreliable.

| E_DRI in percent | Interpretation |
| --- | --- |
| `<10%` | Reliable under the DRI consistency criterion |
| `10–20%`, including exactly 10% and 20% | Use with caution |
| `>20%` | Unacceptable; do not accept the axial estimate |
| Unavailable / `NaN` | No fit-error assessment; no reliability grade |

For finite `E_DRI >=10%`, the warning recommends adjusting the DRI fitting
angle range and repeating the analysis. Recommended limits are
**`low_angle >10`** and **`high_angle <80`**, both in degrees. Current console
messages also use the labels `lowAngle`/`highAngle`; Python keyword arguments
remain `low_angle`/`high_angle`.

Valid bounds are `0 <= low_angle < high_angle <=90`. Recommended limits
produce advice, not automatic setting changes. `num_cones` must be a positive
integer; values below 5 remain accepted. Choose angles from directional curve
quality and sampling, rather than minimizing E_DRI alone.

### Z sampling: manual control and axial interpretation

`z_sampling` is **False by default** and must be enabled manually. Voxel
spacing, anisotropy, and estimated resolution never enable it automatically.
Legacy `None` is treated as `False`.

When enabled, the axial stage uses complementary odd/even Z planes from the
working volumes and doubles the effective Z spacing. This can reduce shared
axial correlations associated with adjacent planes and inter-plane
inconsistencies. It also reduces sampling density and may change directional
crossings and the inferred result. It is separate from spacing-matching
preprocessing.

If axial resolution seems implausibly small for the acquisition, enable
`z_sampling`, rerun, and compare the axial estimate, directional crossings
and E_DRI. The code does not automatically classify an estimate as optimistic.
A finite single-frame axial result with Z sampling OFF receives a conditional
interpretation note.

```python
from src.sFSC import sFSC

# volume is an already loaded (Z,Y,X) array; spacing is [z,y,x] in nm.
result = sFSC(volume, pixel_size=[60, 20, 20], z_sampling=True,
              low_angle=15, high_angle=60)
```

The setting is returned in `result.z_sampling_enabled` and `result.parameters`.
The conditional note is returned in `result.interpretation_note`.

### Warnings and prompts

| Situation | Python behavior |
| --- | --- |
| Extrapolation succeeds | Stage-specific console notice |
| Missing crossing or failed/unsupported extrapolation | `RuntimeWarning`; unavailable estimate remains `NaN` |
| No usable directional crossing or invalid DRI fit | DRI-specific `RuntimeWarning` |
| Finite E_DRI >=10% | `RuntimeWarning` with grading and fitting-angle advice |
| `low_angle <=10` or `high_angle >=80` | `UserWarning` with recommended angle limits |
| Single-frame axial estimate, Z sampling OFF | Conditional console note and `result.interpretation_note` |

sFSC figures omit E_DRI interpretation and other diagnostic text.
The console retains all guidance. Demo exports also retain a guidance text file
and JSON metadata containing the grading and axial note.

### Complete grading guide on every sFSC exit

Every public sFSC call appends this guide to the console when it exits:

```text
E_DRI (relative fit error): <10% reliable; 10-20% use with caution; >20% unacceptable.
```

The `with_sfsc_dri_guidance` wrapper also prints it for unavailable crossings,
failed fits, early returns, and parameter/runtime errors. It is independent of
plotting and of whether E_DRI is high. An unavailable value receives no grade.
The guide stays in textual output rather than adding repetitive CSV columns.


### sFSC results, plotting and saved files

`sFSC(...)` returns a `DirectionalResolutionInference` result. Its compact
summary records singleframe, threshold, extrapolation trigger/stage,
lateral/axial resolution in nm, and E_DRI in percent. Z-sampling status
and interpretation notes remain in the result fields and metadata.
An available lateral/axial resolution profile is always displayed. There is
no `plot` argument or `plot_results` demo setting for sFSC.
`plot_fitting_curve=False` remains the default for the separate, additional
DRI-fit figure. Existing scripts should remove `plot=...` from sFSC calls.
sFSC uses a 3D volume/pair and has no 2D-stack batch switch.

| Content | Filename examples |
| --- | --- |
| Summary / metadata | `sFSC_results.csv`, `sFSC_results.json` |
| Numerical DRI data | `sFSC_DRI-data.npz` |
| Interpretation | `sFSC_E-DRI-guidance.txt` |
| Resolution preview (when available) | `sFSC_lateral-axial-resolution.png` |
| Optional additional fit preview | `sFSC_DRI-elliptical-fit.png` |

## srFRC

Spatially resolved local-resolution mapping for 2D images and stacks.

Configure `RollingFourierRingCorrelation(...)`, then call `.execute(image_stack)`.

### srFRC input and calibration

Provide one `(Y,X)` image or one stack, normally `(T,Y,X)`.
`stack_axis=2` accepts a `(Y,X,T)` stack. Paired mode normally uses
adjacent overlapping frames from this stack; a separate matching
second image/stack remains an optional paired-input form.

### srFRC parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `pixel_size` | Required | Original calibrated scalar XY spacing in nm |
| `single_frame` | `True` | Generate a diagonal pair from each frame; False uses adjacent frames |
| `threshold` | `'3-sigma'` | Only alternative: `'1/7 hard threshold'` |
| `batch_processing` | `True` | Process every frame or adjacent pair |
| `stack_axis` | `0` | Frame axis: 0 for `(T,Y,X)`, 2 for `(Y,X,T)` |
| `plot` | `True` | Preview only the first frame/pair in either batch mode; False suppresses automatic plots |
| `motion_correction` | `False` | Align paired frames before correlation |
| `block_size` | `64` | Local window width; rounded up to an even integer |
| `skip` | `2` | Rolling step, capped at half the window width |
| `adaptive_background` | `True` | Automatic Otsu-based background detection |
| `monitor_change` | `False` | Compare different Otsu coefficients for the first frame/pair; disable batch mode |
| `background_threshold` | `30` | Manual 0–255 intensity threshold, used only when both background switches are False |
| `parallel` | `True` | Enable parallel rolling computation |
| `patch_size` | `64` | Parallel patch width |

The public choices select the **primary** criterion: `'1/7 hard threshold'`
or `'3-sigma'`; srFRC still defaults to `'3-sigma'`. 

### srFRC background settings

`monitor_change=True` compares background thresholds using different Otsu coefficients on the first frame/pair and disables batch processing. Otherwise, use automatic background detection or specify a manual `background_threshold` (0–255).


### srFRC stack batching and plotting

srFRC uses one stack by default. With `batch_processing=True`, an N-frame
single-frame stack produces N results; `single_frame=False` produces N-1
overlapping adjacent pairs: `(1,2)`, `(2,3)`, ..., `(N-1,N)`. Paired mode
needs at least two frames if a separate second input is omitted.

With `batch_processing=False`, only the **first** frame is analyzed.
Paired mode also uses the following frame. A separate matching second
image/stack is retained as an optional paired-input form.

`plot=True` (default) shows only the first frame/pair
regardless of batch mode; batch output still contains all computed results. Paired mode shows the first-and-second-frame calculation.
`plot=False` hides previews; numerical batches and exported data remain complete.
The demos keep numerical output separate from user-generated comparison plots and optional file exports.





### srFRC results and saved files

Each summary row records the frame/pair and background setting, singleframe,
threshold, mean/min/max resolution in nm, rFRC_value, and the final
background_threshold.

| Content | Filename examples |
| --- | --- |
| Summary / metadata | `srFRC_results.csv`, `srFRC_results.json` |
| Final map arrays | `srFRC_results.npz` |
| Final numeric maps | Separate Float32 TIFF files for each background setting |
| Optional per-frame preview | First-frame resolution-map PNG |

Each background setting has separate numeric maps and summaries. TIFF maps retain nm values; actual background thresholds are recorded in the results.

## Demo outputs

The project ships **three `.ipynb` notebooks** in `demos/`. Run them with
JupyterLab or an editor supporting notebooks. They demonstrate the original
sFRC single-/paired-frame comparison, sFSC Z-sampling comparison and srFRC
background/parallel settings.

Plots are implemented in `src/utils/plotting.py` (`ResolutionPlotter`); older plotting helpers remain compatible wrappers.

The `src.utils.save_results` helpers write under **`Path.cwd() / 'results'`**
when invoked; the core numerical entry points do not write files. An example
import is:

```python
from src.utils.save_results import save_sfrc_results
```

| Utility | Output |
| --- | --- |
| `save_sfrc_results(result, plot=True)` | FRC curves, CSV/JSON summary and optional first-result PNG |
| `save_sfsc_results(result, plot_fitting_curve=False)` | 3D resolution/DRI data, guidance and available PNG previews |
| `save_srfrc_results(analysis, plot=True)` | Numeric maps (float32 TIFF/NPZ), CSV/JSON summary and optional first-result PNG |

| Example file | Meaning |
| --- | --- |
| `sFRC_results.csv` | Global 2D resolution (nm) by frame/pair |
| `sFSC_results.csv` | Lateral/axial resolution (nm) and E_DRI (%) |
| srFRC numeric TIFFs | Float32 resolution maps for the selected background settings |

Saved TIFFs retain numeric nm values; `0` indicates background/invalid local
estimation rather than 0 nm resolution. Existing named files may be overwritten
by repeated runs. File export is independent of plot display.
