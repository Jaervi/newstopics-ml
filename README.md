# Quick guide

## Prerequisites
1. Ensure Python is installed on your system.
2. Set the following environment variables:
   - `RAW_DATA_FOLDER_PATH`: Path to the root folder containing raw JSON data.
   - `PROCESSED_DATA_DEST_PATH`: Path to the destination folder for processed CSV files.

### Virtual Environment Setup

It is recommended to use a virtual environment to manage dependencies for this project. Follow these steps:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows (PowerShell):
     ```bash
     .\venv\Scripts\Activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Project setup

1. Set the required environment variables in .env:
   ```bash
   RAW_DATA_FOLDER_PATH = "<path-to-raw-data>"
   PROCESSED_DATA_DEST_PATH = "<path-to-destination-folder>"
   ```

2. Run the script to process all raw data:
   ```bash
   python src/process_raw_dataset.py
   ```


## Notes
- The `process_json_into_csv` function converts JSON objects into CSV rows with the following fields:
  - `headline`
  - `date` (formatted as YYMM)
  - `topic`
  - `organization`
  - `fine_topic`

## Explore data
- You can run a small exploration script with
   ```bash
   python src/explore_dataset.py
   ```