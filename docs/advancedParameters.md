# Advanced parameters and result interpretation — Python

Return to the [README](../README.md).

This guide applies to this Python repository. The entry points are `sFRC`,
`sFSC`, and `RollingFourierRingCorrelation` in `src/`. They accept image
arrays and return results in memory, with optional display controls. Demos
read the example TIFFs and invoke plotting/export utilities from `utils/`.
The core analysis functions never save files or take output-path arguments.

Go directly to the analysis you are using:

- [sFRC — global 2D resolution](#sfrc)
- [sFSC — lateral/axial 3D resolution and DRI](#sfsc)
- [srFRC — local resolution maps](#srfrc)

Use the original acquisition calibration. Input sampling, reported resolutions, and srFRC map values are all in **nm**.

`pixel_size` must be supplied with the acquisition calibration. sFRC and
sFSC use `None` as a placeholder, not a usable spacing default.

## Threshold choices and auto plotting

All three entry points accept exactly two correlation criteria:
`threshold='1/7 hard threshold'` and `threshold='3-sigma'`. sFRC/sFSC default
to the first; srFRC defaults to the second. For existing scripts, `fixed_threshold`
and `1/7` remain aliases of the same fixed criterion. Curve objects and compact
summary/parameter dictionaries retain the internal key `fixed_threshold`; these
aliases are not additional threshold types. 2-sigma/5-sigma are not public choices.

For sFRC/srFRC, `plot='auto'` and `plot=True` display only the result selected
by the zero-based `current_frame`, independently of batch processing:

| batch_processing | Computation | Display |
| --- | --- | --- |
| `True` (default) | All frames or adjacent pairs | Only current_frame or its adjacent pair |
| `False` | Only current_frame or its adjacent pair | That same one result |

Single-frame mode displays the current frame. Paired mode displays one result
for current_frame and the next frame. `plot=False` hides figures. Text
'true'/'false' is also accepted. With separate paired arrays, current_frame
selects corresponding frames from both arrays. sFSC treats its input as a 3D
volume and always displays an available resolution result. sFSC has no public
`plot` argument or demo `plot_results` switch; `plot_fitting_curve` controls
only the extra DRI-fit figure.

The save utilities write numerical data and summary rows for **all computed
frames**; optional PNG previews follow the current-frame rule. Monitoring retains
the background comparisons for this same frame/pair. If the current frame has
no following frame in paired mode, no other pair is substituted: a console
notice explains the unavailable preview and the full computed batch is retained.

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
| `current_frame` | `0` | Zero-based preview frame; also selects computation when batching is False |
| `stack_axis` | `0` | Frame axis: 0 for `(T,Y,X)`, 2 for `(Y,X,T)` |
| `plot` | `'auto'` | Display only current_frame, independently of batching; False hides plots |
| `motion_correction` | `False` | Align paired frames before correlation |

### sFRC curve extrapolation

Resolution normally uses a measured crossing of the correlation curve and
the selected threshold. When no usable crossing is available, automatic
curve extrapolation is attempted for `threshold='fixed_threshold'`.
The fitted crossing lies beyond the measured frequency range and adds no
measured information. Extrapolation is not available for `3-sigma`.

`curve_extrapolation` is an automatic procedure, not a public parameter.
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

With `batch_processing=False`, `current_frame` selects one **zero-based**
frame. Paired mode also uses the following frame. A separate matching second
image/stack is retained as an optional paired-input form.

`plot='auto'` and `plot=True` show only the `current_frame` result with
batching ON or OFF. Paired mode shows the current-and-next-frame calculation.
`plot=False` hides previews; numerical batches and exported data remain complete.
The demos use lowercase plot_results/save_results controls and keep saving separate.

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
| `xy_mode` | `'slice'` | Lateral estimation from slices; `'3D'` uses the central Fourier plane |
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

### sFSC results, plotting and saved files

`sFSC(...)` returns an `EllipticalResolutionFitting` result. Its compact
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
| `current_frame` | `0` | Zero-based preview frame; also selects computation when batching is False |
| `stack_axis` | `0` | Frame axis: 0 for `(T,Y,X)`, 2 for `(Y,X,T)` |
| `plot` | `'auto'` | Display only current_frame, independently of batching; False hides maps |
| `motion_correction` | `False` | Align paired frames before correlation |
| `block_size` | `64` | Local window width; rounded up to an even integer |
| `skip` | `2` | Rolling step, capped at half the window width |
| `adaptive_background` | `True` | Use 1.2 times Otsu; forced False by monitor_change |
| `monitor_change` | `False` | Compare Otsu multipliers 0.8, 1.0, and 1.2; takes priority |
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

`monitor_change=True` has priority and **forces `adaptive_background=False`**.
This is a comparison mode, not a request for a single manual threshold.

| monitor_change | adaptive_background (effective) | Background thresholds | background_threshold |
| --- | --- | --- | --- |
| `True` | `False`, forced | `0.8 x Otsu`, `1.0 x Otsu`, `1.2 x Otsu` | Ignored |
| `False` | `True` | `1.2 x Otsu` | Ignored |
| `False` | `False` | Fixed manual value | Used, default `30` |

Both requested values of adaptive_background produce the same three comparisons
when monitoring is True. A console notice explains the override; effective settings
are stored in `analysis.parameters`. Three distinct map groups (`bg0.8`, `bg1.0`,
`bg1.2`) and corresponding per-frame summary rows are retained. For a normal
automatic run use `monitor_change=False, adaptive_background=True`; for a manual
run use `monitor_change=False, adaptive_background=False, background_threshold=30`.

Otsu is evaluated for each working image pair. Manual and final reported intensity
thresholds use the **0-255 scale**, independently of the normalized float32 image
values. Correlation threshold choices never alter these intensity-scale units.

### srFRC stack batching and plotting

srFRC uses one stack by default. With `batch_processing=True`, an N-frame
single-frame stack produces N results; `single_frame=False` produces N-1
overlapping adjacent pairs: `(1,2)`, `(2,3)`, ..., `(N-1,N)`. Paired mode
needs at least two frames if a separate second input is omitted.

With `batch_processing=False`, `current_frame` selects one **zero-based**
frame. Paired mode also uses the following frame. A separate matching second
image/stack is retained as an optional paired-input form.

`plot='auto'` and `plot=True` show only the `current_frame` result with
batching ON or OFF. Paired mode shows the current-and-next-frame calculation.
`plot=False` hides previews; numerical batches and exported data remain complete.
The demos use lowercase plot_results/save_results controls and keep saving separate.

### srFRC monitored outputs and sjet comparison

Monitoring keeps `bg0.8`, `bg1.0` and `bg1.2` as three separate groups. Previous
maps and summary dictionaries retained them, but several convenience methods
returned only the last group. They now return all monitored groups by default:

```python
maps = analysis.get_rfrc_map()          # dict with bg0.8, bg1.0, bg1.2
statistics = analysis.calculate_global()  # same keys; native-grid statistics
panels = analysis.get_panels()         # one quality array per background group
map08 = analysis.get_rfrc_map('bg0.8')  # select one group's array
```

One-background runs keep their original NumPy-array return types. Monitoring
returns dictionaries; callers needing an array should select a background label.
The low-level iterate method likewise returns all monitored maps rather than
overwriting its return value on each pass. Compatibility attributes rfrc_map,
mean_res, min_res, max_res and rfrc_value still describe the last frame/group;
the accessors above provide complete results. No statistics are recomputed after
interpolation.

All groups finish computing before the optional comparison is shown. The current
frame/pair is displayed side by side for all groups, in one window. This avoids
blocking on the first figure before later groups have finished. Every comparison
axis and individual PNG uses sJet and the same display range. No warning text is
embedded in the plots. Numerical results and per-frame summaries retain the full
requested batch.

Saving produces three distinct TIFFs and per-group CSVs named with bg0.8, bg1.0
and bg1.2; optional current-frame PNGs retain those same group names. The combined
CSV/JSON/NPZ exports remain available. Plotting and saving helpers remain in utils/.

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
| Final numeric maps | `srFRC_bg1.2_resolution-map.tif`, `srFRC_manual-bg_resolution-map.tif` |
| Optional per-frame preview | `srFRC_frame0001_bg1.2_resolution-map.png` |

Frame identifiers use `frame0001`; pairs use `pair0001-0002`. Each background
setting has a `(T,Y,X)` map array and a float32 TIFF with nm units, without
normalization or colour-LUT conversion. The `bg1.2` label is an Otsu multiplier;
the actual intensity threshold is stored in `background_threshold` and
`final_background_thresholds`. Manual background uses `manual-bg`.
TIFF, NPZ and summaries use the same final restored maps.

## Demo outputs

Shared output-folder and utility conventions for the three demos:

The demos directly read explicit paths under project-root `example_data/`.
They call the utilities only when saving is enabled. **`Path.cwd()/'results'`**
is the output folder, independent of the input file's location.

The `demos/` folder contains only `sFRC_demo.py`, `sFSC_demo.py` and
`srFRC_demo.py`. Plot/export helpers are in **`utils/save_results.py`**; use
`from utils.save_results import ...` in demos, tests and user scripts:

| Utility | Role |
| --- | --- |
| `save_sfrc_results(result, plot=True)` | Compact summary, numerical curves, optional PNG previews |
| `save_sfsc_results(result, plot_fitting_curve=False)` | Summary, DRI data, E_DRI guidance and an available resolution preview; optional extra fit PNG |
| `save_srfrc_results(analysis, plot=True)` | Final maps, summary/metadata, float32 TIFF and optional previews |

sFRC/srFRC numerical exports do not require plotting and retain the current-frame
preview rule. The sFSC save helper has no `plot` switch and includes the
resolution PNG whenever the estimate is available. File writing remains an
optional demo step controlled by `save_results`. No save switch, output directory, or base-name
parameter is added to the core API.

See the result/file subsection of each analysis above for its filenames.
Repeated runs overwrite matching filenames and leave other result files intact.
