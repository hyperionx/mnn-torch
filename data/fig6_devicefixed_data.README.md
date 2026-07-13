# Static MNIST training archive

`fig6_devicefixed_data.json` is the lightweight numerical archive used for the
static MSNN and MCSNN training-history figures. Its SHA-256 digest is
`60b36f0e11c92fdd3ee09a6a6803419803f6c77ffcf1e12b100ec0db56422306`.

The top-level `config` object records the seeds, epoch count, model dimensions,
optimizer settings, dataset subsets, device, and the three tested conditions.
The `fc` and `conv` objects each contain `ideal`, `memristive_pf`, and `fault`
condition objects. Every condition has:

- `acc_hist`: test accuracy in percent, shaped `[seed, epoch]`;
- `loss_hist`: test cross-entropy loss, shaped `[seed, epoch]`.

For this archive both arrays have shape `[3, 12]`, corresponding to seeds
`[0, 1, 2]`. The publication plotting function uses the per-epoch arithmetic
mean and population standard deviation (`numpy.std(..., ddof=0)`) across those
three recorded runs. It does not bootstrap, expand, or synthesize observations.
`wall_seconds` records elapsed run time and is not plotted.

The optional reduced validation in `experiments/01_device_and_static.ipynb`
uses the canonical archive generator in that notebook. The generator constructs
both MSNN and MCSNN models, runs all three conditions, and records both metrics
in this same schema. Its `publication` budget exactly matches the configuration
above; `reduced` and `smoke` select smaller deterministic budgets. Generation is
opt-in through `MNN_GENERATE_STATIC_ARCHIVE=1`, and a regenerated archive is a
new validation run rather than a replacement for this committed record.
