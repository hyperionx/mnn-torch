# Reduced homeostasis dataset fixture

`homeostasis_reduced_datasets.npz` contains deterministic, class-balanced
subsets of genuine MNIST and Fashion-MNIST images for the live reduced
architecture-dependence experiment in `experiments/02_homeostasis.ipynb`.

- MNIST: 1,000 training and 500 test images.
- Fashion-MNIST: 1,000 training and 500 test images.
- Images retain their original 28 x 28 unsigned-byte values and are divided by
  255 only when loaded.
- Original dataset indices, class labels, selection seed, source tensor hashes,
  and the torchvision version are stored inside the archive.
- Moving-MNIST clips are generated live by translating selected MNIST images
  with zero-filled boundaries; no generated clip is stored here.

Fashion-MNIST is distributed under the MIT licence. MNIST and Fashion-MNIST
remain attributed to their original authors; this fixture is included solely to
make the small offline reproduction deterministic and independently runnable.

SHA-256:
`5efbda60d371596f11fdbc0cff7f240509621c53f05aa6fb37792e729aacd05c`

## Reduced temporal dataset fixture

`temporal_reduced_datasets.npz` contains genuine samples selected from the
official N-MNIST, Spiking Heidelberg Digits (SHD), and IBM DVS Gesture test
distributions. It supports live reduced validation in
`experiments/03_temporal_memory.ipynb` without shipping the multi-gigabyte
source datasets.

- N-MNIST: 20 samples per class, framed into six bins at 2 x 34 x 34.
- SHD: 20 samples per class, framed into six bins across 700 input channels.
- DVS Gesture: 20 samples per class, framed into six bins and spatially reduced
  from 2 x 128 x 128 to 2 x 32 x 32.
- Each benchmark uses a deterministic nonstandard reduced repartition: 12
  samples per class for training and eight per class for testing.
- Original sample indices, source archive hashes, selection seed, framing
  configuration, and Tonic version are embedded in the archive.

These scores are reduced sample-validation results, not standard benchmark test
scores. Exact manuscript scores require the optional provenance-complete
full-sweep archive.

SHA-256:
`0bff22c766f348392a5e4fd8d2c77ac2694ef16f7603dd9c271065e0fd435277`

## Reduced representation dataset fixture

`representations_reduced_datasets.npz` supports the live conv2 workflows in
`experiments/04_representations.ipynb` without requiring a network connection.

- N-MNIST contains 60 genuine test-distribution samples per digit, framed into
  ten bins at 2 x 34 x 34. The reduced run uses a nonstandard 40/20-per-class
  train/test repartition.
- EEG contains 64 genuine MNE sample auditory/visual epochs per class, spanning
  60 EEG channels and 70 samples after 100 Hz resampling. The reduced run uses
  48/16 samples per class for training/testing; the appendix-style embedding
  diagnostic visualizes both partitions and is explicitly in-sample.
- Metadata embedded in the archive records original indices, event identifiers,
  channel names, preprocessing, upstream file hashes, and library versions.

These compact samples support mechanism validation, not numerical replacement
of the full published feature archive.

SHA-256:
`a6303dd229e0d0e8e3ac334f59473f81f2677952b2b62daa3755c049c7aeefbb`
