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