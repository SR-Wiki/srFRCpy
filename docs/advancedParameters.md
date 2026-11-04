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

## Correlation thresholds and automatic preview

All three entry points accept exactly two correlation criteria:
`threshold='1/7 hard threshold'` and `threshold='3-sigma'`. sFRC/sFSC default
to the first; srFRC defaults to the second. For existing scripts, `fixed_threshold`
and `1/7` remain aliases of the same fixed criterion. Curve objects and compact
summary/parameter dictionaries retain the internal key `fixed_threshold`; these
aliases are not additional threshold types. 2-sigma/5-sigma are not public choices.

For sFRC/srFRC, `plot=True` (default) previews **only the first frame or
adjacent pair**; `plot=False` disables automatic plots but never changes the
numerical calculations. No `current_frame` parameter exists. The historical
string spelling `plot='auto'` is still accepted by the implementation for older
scripts, but is not needed in new code.

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

## Input precision, normalization and plot appearance

The demos convert TIFF data directly to **float32** and divide the complete
loaded image, stack or volume by its maximum. The public numerical entry points
apply the same preparation to directly supplied arrays, before any diagonal
averaging or motion correction. Input arrays are copied, so caller data are not
modified. All-zero inputs stay zero; non-finite intensities are rejected with
an explanatory error. There is no intermediate 8-bit/16-bit conversion.

The reading/preparation stage is 32-bit floating point; subsequent FFT, fitting
or interpolation routines may retain their existing higher calculation precision.

One maximum is shared by a complete stack or 3D volume, preserving relative
slice intensities. Separately supplied paired acquisitions are normalized
independently. srFRC also normalizes each generated pair member to a maximum of
one; rolling blocks retain their existing local normalization. Manual and
reported background thresholds retain their **0-255 units**: background tests
convert normalized intensity to that scale, and the automatic Otsu threshold is
reported on that scale. This is an intensity convention, not an 8-bit image cast.

Figures show curves, axes, legends and numerical results. E_DRI grades, fitting
angle advice, extrapolation notices and Z-sampling reminders appear in the
console rather than being embedded in result plots or PNG previews. Existing
text/JSON guidance exports remain available. Correlation curves, the fitted 3D
anisotropic resolution profile and DRI-fit lines use light red **#E67373**
(RGB 230,115,115); simultaneous directional curves use distinguishable red
shades/line styles. Thresholds and measured-point markers retain contrast.

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

1. Lateral sFSC use measured crossings when available. Otherwise,
   automatic extrapolation is attempted for `threshold='fixed_threshold'`.
2. Axial sFSC/DRI first evaluates every directional curve without extrapolation.
   If any direction has a usable measured crossing, only usable measured
   directions enter the fit. Missing directions are excluded.
3. Directional extrapolation is attempted only when **all directions lack
   usable measured crossings**, and only for the fixed 1/7 threshold.
4. Failed extrapolation, an unsupported threshold, or an unusable fit leaves
   unavailable estimates as `NaN` and issues an explanatory warning. An
   unavailable lateral anchor also prevents an axial estimate.

Successful extrapolation prints a stage-specific notice. Failures and
unsupported thresholds use `RuntimeWarning`; the selected threshold is never
silently changed. Internal measured-only controls support the first DRI pass
and rolling FRC and are not public switches.

The sFSC summary records `extrapolation_triggered` as an attempt flag, including
failed attempts. It also records `extrapolation_stage`: `none`, `lateral`,
`axial`, or `lateral and axial`. Lateral tracking includes candidate slices;
the flag alone does not prove that the final selected curve was extrapolated.
Inspect the notices and returned curve/fit information.

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

`result.goodness_of_fit` is a fraction: `0.10` means 10%.
`result.summary['E_DRI_percent']` is already in percent. Estimates above 20%
remain available for inspection/export together with their warning; export
does not make them acceptable.

### Z sampling: manual control and axial interpretation

`z_sampling` is **False by default** and must be enabled manually. Voxel
spacing, anisotropy, and estimated resolution never enable it automatically.
Legacy `None` is treated as `False`.

When enabled, the axial stage uses complementary odd/even Z planes from the
working volumes and doubles the effective Z spacing. This can reduce shared
axial correlations associated with adjacent planes and inter-plane
inconsistencies. It also reduces sampling density and may change directional
crossings and the inferred result. It is separate from spacing-matching
preprocessing and from srFRC cubic-spline map restoration.

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

### sFSC numerical timing

Each call returns `computation_time_seconds` (total numerical time, excluding
its final Matplotlib profile/DRI-fit displays), and `stage_elapsed_seconds`
with **only** `lateral` and `axial` (seconds). `lateral` includes input
normalization/pair preparation and lateral FRC; `axial` includes Z sampling,
resampling, FFT and directional FSC/DRI. Progress display overhead within the
calculation is included. If lateral resolution is unavailable, `axial` is zero.
The two stage values cover a single call, not the notebook-level two-run comparison.

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
| `adaptive_background` | `True` | Use **0.8 × Otsu**; ignored when monitoring |
| `monitor_change` | `False` | Compare **0.6, 0.8 and 1.0 × Otsu** on the first frame/pair; force non-batch mode |
| `background_threshold` | `30` | Manual 0–255 intensity threshold, used only when both background switches are False |
| `parallel` | `True` | Enable parallel rolling computation |
| `patch_size` | `64` | Parallel patch width |

The public choices select the **primary** criterion: `'1/7 hard threshold'`
or `'3-sigma'`; srFRC still defaults to `'3-sigma'`. Both serial and parallel
paths retain the existing fallback order: **primary, then 5-sigma x 0.6, then
2-sigma x 1.5**. A fallback is used only when earlier criteria have no finite
positive estimate. Zero is used only if all criteria fail. Summary threshold
fields identify the selected primary criterion. Padding calibration is retained
for sigma thresholds; the hard cutoff stays exactly 1/7. Internal fallback
criteria are not extra public choices. Local curves are not extrapolated.
Correlation threshold is independent of the background-intensity controls.

### srFRC background parameter relationships

`monitor_change=True` has the highest priority: it sets
`adaptive_background=False` **and** `batch_processing=False`, so only the first
frame/pair is analyzed against three automatic background thresholds.

| `monitor_change` | Effective `adaptive_background` | Effective `batch_processing` | Background cutoff(s), on 0–255 intensity scale |
| --- | --- | --- | --- |
| `True` | `False` | `False` | `0.6 × Otsu`, `0.8 × Otsu`, `1.0 × Otsu` |
| `False` | `True` | User setting | `0.8 × Otsu` |
| `False` | `False` | User setting | Manual `background_threshold` (default `30`) |

The internal group names (`bg0.6`, `bg0.8`, `bg1.0`) refer to Otsu multipliers,
**not** actual thresholds. The absolute computed cutoff for each frame/pair is
reported as `background_threshold` in the results (0–255 units). Otsu is
evaluated for each working pair; FRC correlation thresholds are independent of
these background-intensity cutoffs.

For regular automatic mapping, use the defaults. For manual masking, use
`monitor_change=False, adaptive_background=False` and specify
`background_threshold`.

### srFRC numerical timing

Existing timing is unchanged: `computation_time_seconds` (alias
`elapsed_seconds`), `background_elapsed_seconds`, and
`frame_elapsed_seconds_by_background` are measured before any optional map
plotting or export. Background and frame timings are nested in the total and
must not be added to it.

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

### srFRC monitored outputs and sJet comparison

In monitoring mode, all three background groups are retained separately. To
access the computed maps or statistics, select a group if an array is required:

```python
maps = analysis.get_rfrc_map()           # dict: bg0.6, bg0.8, bg1.0
stats = analysis.calculate_global()      # native-grid statistics by group
map_08 = analysis.get_rfrc_map('bg0.8')   # one numeric map in nm
```

The first frame/pair is previewed, with three maps shown side by side using
sJet. Each group's **absolute** background threshold is reported in results;
`bg0.8` is only an internal/export group ID. A single-background run returns
its original NumPy array. Monitoring may return dictionaries from the
convenience accessors; `rfrc_map` and other legacy scalar attributes refer to
the last result/group and should not be used to enumerate monitored groups.

The export helper can write separate TIFF/CSV files for `bg0.6`, `bg0.8`, and
`bg1.0`, alongside combined CSV/JSON/NPZ results. In a normal automatic run,
the numeric map uses the `bg0.8` group label. All output maps retain original
XY dimensions; their numerical values are not encoded as PNG colours.

### srFRC map restoration

Single-frame srFRC restores the reduced map to the original XY size with a
**separable cubic spline**, using `scipy.interpolate.CubicSpline` along X and
then Y, with not-a-knot boundary conditions. Fourier map restoration and the
`map_interpolation` selection parameter have been removed. Backend metadata
records `cubic_spline`; paired maps are already full size and record `none`.

Zero-based output index j samples source coordinate j/2. Coordinates beyond the
last source sample use that edge value, including odd original dimensions.
Singleton axes are constant; axes with two or three samples use the corresponding
linear or quadratic interpolation. Existing source samples are retained at even
output coordinates. This is an interpolating cubic spline, not bicubic convolution.

The original finite positive foreground mask is repeated to the output grid;
it is never inferred from interpolated values. Points outside this support remain
zero. Non-positive candidates become zero; finite positive candidates are bounded
by the positive source minimum and maximum after source-map cleanup below.
These guards prevent interpolation artifacts from becoming extra foreground.
New spline samples can change the foreground mean; exact distribution or mean
preservation is not guaranteed or imposed.

**All resolution statistics are calculated BEFORE upsampling.** The native
rolling map is first cleaned: non-positive samples become background zero and
finite positive samples are at least **2.3 times the original pixel size in nm**.
Mean/min/max resolution and `rFRC_value=mean/min-1` are then calculated from its
finite positive foreground. Background zeros are excluded; an empty foreground
gives `NaN` statistics. Only afterwards is the map enlarged for display/export.
The spline is bounded by the cleaned source range, so restored positive values
also retain the required lower bound. The doubled diagonal-pair spacing is used
for correlation frequencies, not for a second map floor.

`statistics_stage` records `pre_upsampling`, and `statistics_map_shape` records
the native grid dimensions. `calculate_global()` returns the cached native-grid
statistics; it never recalculates them from `rfrc_map`. CSV/JSON summaries use
these same cached values. TIFF/NPZ/PNG continue to contain original-size maps;
their spatial mean can differ from the reported native-grid mean.

### srFRC results and saved files

Each summary row records the frame/pair and background setting, singleframe,
threshold, mean/min/max resolution in nm, rFRC_value, and the final
background_threshold.

| Content | Filename examples |
| --- | --- |
| Summary / metadata | `srFRC_results.csv`, `srFRC_results.json` |
| Final map arrays | `srFRC_results.npz` |
| Final numeric maps | `srFRC_bg0.8_resolution-map.tif`, `srFRC_manual-bg_resolution-map.tif` |
| Optional per-frame preview | `srFRC_frame0001_bg0.8_resolution-map.png` |

Frame identifiers use `frame0001`; pairs use `pair0001-0002`. Each background
setting has a `(T,Y,X)` map array and a float32 TIFF with nm units, without
normalization or colour-LUT conversion. The `bg0.8` label is an Otsu multiplier;
the actual intensity threshold is stored in `background_threshold` and
`final_background_thresholds`. Manual background uses `manual-bg`.
TIFF, NPZ and summaries use the same final restored maps.

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
| `srFRC_bg0.8_resolution-map.tif` | Float32 numeric resolution map for the default automatic background |
| `srFRC_bg0.6_resolution-map.tif` | One group from a monitored run |

Saved TIFFs retain numeric nm values; `0` indicates background/invalid local
estimation rather than 0 nm resolution. Existing named files may be overwritten
by repeated runs. File export is independent of plot display.
