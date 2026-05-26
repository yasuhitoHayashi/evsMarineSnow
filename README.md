# evsMarineSnow

Analysis code for the evsMarineSnow manuscript.

## Repository Layout

```text
tracking/          event-based particle tracking
sizeEstimation/    particle-size estimation from tracked EVS events
samplingAnalysis/  reference PSD and frame-sampling analysis
mainFigs/          figure scripts and exported figures
```

## Python environment
The Python scripts use NumPy, SciPy, and pandas.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Convert RAW files to CSV
Convert Prophesee RAW files to event CSV files with `metavision_file_to_csv`.

```bash
metavision_file_to_csv -i input.raw
```

## Build the evsMarineSnow tracker
The tracker is a C++17 command-line program built with CMake.

```bash
cmake -S tracking/cpp -B tracking/cpp/build
cmake --build tracking/cpp/build --target tracker_cli
```

The executable is written to:

```text
tracking/cpp/build/tracker_cli
```

## Run tracking
Run the tracker on a directory of event CSV files.

```bash
bash tracking/evsMarineSnow.sh path/to/event_csv_directory
```
For each CSV file, an .ndjson file is written in the same directory.
The tracker uses positive-polarity events.

## Run size estimation
Estimate projected particle area from each tracking NDJSON file.

```bash
python sizeEstimation/saveSize.py \
  -i path/to/events.ndjson \
  --exclude-entering-tracks
```

If no output path is given, `saveSize.py` writes `*_size.csv` next to the input
NDJSON. The output includes projected area in pixels and equivalent circular
diameter in millimeters.

To process all NDJSON files under a directory:

```bash
bash sizeEstimation/saveSize.sh path/to/tracking_output
```

## Run sampling analysis
Build the EVS reference PSD and the virtual frame-sampling detections from
`*_size.csv`.

```bash
python samplingAnalysis/build_reference_psd.py -i path/to/events_size.csv
python samplingAnalysis/particle_frame_detections.py -i path/to/events_size.csv
```

This creates:

```text
events_reference_psd.csv
events_frameSampling.csv
```

Then build the tables used by the figure scripts.

```bash
python samplingAnalysis/build_phase_bin_counts.py \
  -i path/to/events_frameSampling.csv \
  -o samplingAnalysis/data_output/cam85M/tables/phase_bin_counts_table.csv

python samplingAnalysis/build_bulk_ratio_tables.py \
  -i path/to/frame_sampling_directory \
  -o samplingAnalysis/data_output/bulk_ratio_alltimes \
  --recursive
```

`build_phase_bin_counts.py` aggregates detections by particle-size bin, frame
rate, and phase. `build_bulk_ratio_tables.py` builds frame-to-EVS ratio tables
for number density and volume density.