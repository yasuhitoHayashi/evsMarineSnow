# evsMarineSnow
repository for evsMarineSnow paper

## Convert RAW files to CSV
Event CSV files must be generated from Prophesee RAW files using `metavision_file_to_csv`:
```bash
metavision_file_to_csv -i input.raw
```

## Build the evsMarineSnow tracker
The evsMarineSnow tracker is a C++17 command-line program built with CMake.

```bash
cmake -S tracking/cpp -B tracking/cpp/build
cmake --build tracking/cpp/build --target tracker_cli
```

This creates:

```text
tracking/cpp/build/tracker_cli
```

## Run tracking
After building `tracker_cli`, run the tracking script with a directory containing event CSV files:

```bash
bash tracking/evsMarineSnow.sh path/to/event_csv_directory
```
For each CSV file, an .ndjson file is written in the same directory.
evsMarineSnow tracker uses positive-polarity events and starts the time window from the first
accepted event in each file.

## Run size estimation
After tracking, estimate projected particle area from each tracking NDJSON file:

```bash
PYENV_VERSION=evsMarineSnow pyenv exec python sizeEstimation/saveSize.py \
  -i path/to/events.ndjson \
  --exclude-entering-tracks
```

If no output path is given, `saveSize.py` writes `*_size.csv` next to the input
NDJSON. The estimator uses the first half of each track, a quadratic trajectory
model, 1 ms area windows, and filled occupancy area in pixel-squared units.

To process all NDJSON files under a directory:

```bash
PYTHON_BIN="pyenv exec python" PYENV_VERSION=evsMarineSnow \
  bash sizeEstimation/saveSize.sh path/to/tracking_output
```