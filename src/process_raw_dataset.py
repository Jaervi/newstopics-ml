import os
import json
import pandas as pd
import csv
from dotenv import load_dotenv

# Resolve .env path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # one level up from src/
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)


RAW_DATA_FOLDER_PATH = os.getenv("RAW_DATA_FOLDER_PATH")
PROCESSED_DATA_DEST_PATH = os.getenv("PROCESSED_DATA_DEST_PATH")

if not RAW_DATA_FOLDER_PATH or not PROCESSED_DATA_DEST_PATH:
    raise ValueError("Environment variables RAW_DATA_FOLDER_PATH and PROCESSED_DATA_DEST_PATH must be set.")
save_errors = input("Do you want to save error logs to a file? (y/n, default 'n'): ").strip().lower()
if save_errors == "y":
    save_errors = True
else:
    save_errors = False

#Extracts the organization name following these two formats
# 60-13211-UA-talousjaarki
# 1343-pohjanmaa
# And lists with ; separator
def extract_organization(org_field: str) -> str:

    if not org_field:
        return ""
    org = org_field.split(";")[0]  # Take only the first organization

    parts = org.split("-")
    # As there are no two letter orgs, we can use this heuristic

    if len(parts) > 2 and len(parts[2]) == 2:
        # Take everything after the third dash
        return "-".join(parts[3:]).strip()
    elif len(parts) >= 2:
        # If the third part is not a two-letter code, assume its a two part format
        return "-".join(parts[1:]).strip()
    else:
        return ""
    


def process_raw_data():
    # Placeholder for the actual data processing logic
    filename = input("Enter a name for the output CSV file (without extension): ").strip()
    if not filename:
        filename = "dataset"
    print(f"Processing raw data from {RAW_DATA_FOLDER_PATH} and saving to {PROCESSED_DATA_DEST_PATH} into file {filename}.csv")
    all_rows = []
    counter = 0
    total_files = sum(len(files) for _, _, files in os.walk(RAW_DATA_FOLDER_PATH) if files)

    for root, _, files in os.walk(RAW_DATA_FOLDER_PATH):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                rows = process_single_file(file_path)
                all_rows.extend(rows)
                counter += 1
                print(f"Processed {counter}/{total_files} files, total rows: {len(all_rows)}\r", end="")


    # Save combined CSV
    os.makedirs(PROCESSED_DATA_DEST_PATH, exist_ok=True)
    dest_file = os.path.join(PROCESSED_DATA_DEST_PATH, f"{filename}.csv")

    df = pd.DataFrame(all_rows, columns=["headline", "date", "topic", "organization", "fine_topic", "year", "month"])
    df.to_csv(dest_file, index=False, encoding="utf-8")

    print(f"Processed {len(all_rows)} rows into {dest_file}")


def log_unprocessed_file(file_path, reason):
    """Logs the reason for not processing a file into a metadata CSV."""
    metadata_file = os.path.join("metadata", "unprocessed_files.csv")
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

    with open(metadata_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([file_path, reason])


def process_single_file(file_path):
    rows = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            if not isinstance(data, dict) or "data" not in data:
                print(f"Skipping {file_path}, no 'data' array found")
                return rows

            for obj in data["data"]:
                row = process_json_into_csv_row(obj)
                if row:
                    rows.append(row)
    except Exception as e:
        if save_errors:
          log_unprocessed_file(file_path, str(e))
        print(f"Error processing {file_path}: {e}")
    return rows

# Gets a single json entry as parameter. Returns a csv string of the object split into fields in an array.
# - headline
# - date
# - topic
def process_json_into_csv_row(json_obj):
    try:
        analytics_obj = json_obj["meta"]["analytics"]
    except (KeyError, TypeError):
        if save_errors:
          log_unprocessed_file("some json obj", "Missing 'meta' or 'analytics' in JSON object")
        return None
        


    headline = analytics_obj.get("yle_stickytitle", "").replace("_", " ").replace(",", "")
    date = analytics_obj.get("yle_pub", "").replace("-", "")[2:6]  # YYMM
    topic = analytics_obj.get("countername", "").split(".")[0]
    organization = extract_organization(analytics_obj.get("yle_organization", ""))
    fine_topic = analytics_obj.get("yle_topic", "").split(";")[0]

    year = date[:2]
    month = date[2:4]

    #organization = "".join(analytics_obj.get("yle_organization", "").split(";")[0].split("-")[3:]).strip()
    #topic = analytics_obj.get("yle_topic", "").split(";")[0]

    if not headline or not date or not topic: #Requiring headline, date, topic but not others
        if save_errors:
          missing_fields = []
          if not headline:
              missing_fields.append("headline")
          if not date:
              missing_fields.append("date")
          if not topic:
              missing_fields.append("topic")
          if not organization:
              missing_fields.append(f"organization '{analytics_obj.get('yle_organization', '')}'")
          if not fine_topic:
              missing_fields.append(f"fine_topic '{analytics_obj.get('yle_topic', '')}'")

          if missing_fields:
              log_unprocessed_file("some json obj", f"Missing fields: {', '.join(missing_fields)}")
              return None
        return None

    return [headline, date, topic, organization, fine_topic, year, month]



if __name__ == "__main__":
    process_raw_data()