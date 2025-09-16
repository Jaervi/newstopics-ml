import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Resolve .env path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

DATA_PATH = os.getenv("PROCESSED_DATA_DEST_PATH")

# Load dataset
#df = pd.read_csv(os.path.join(DATA_PATH, "dataset.csv"))
filename = input("Enter the name of the CSV file to explore (without extension, default 'dataset'): ").strip()
save_files = input("Do you want to save the frequency counts to files? (y/n, default 'n'): ").strip().lower()
if save_files not in ['y', 'n']:
    save_files = 'n'
if not filename:
    filename = "dataset"
df = pd.read_csv(os.path.join(DATA_PATH, f"{filename}.csv"), dtype={'date': str})


# Quick look at the first rows
print("First 5 rows:")
print(df.head(), "\n")

# Basic info
print("Dataset info:")
print(df.info(), "\n")

# Check for missing values
print("Missing values per column:")
print(df.isna().sum(), "\n")

# Frequency counts of 'topic'
print("Topic frequencies:")
topic_counts = df['topic'].value_counts()
#Save topic counts into file
os.makedirs("metadata", exist_ok=True)

# Ask the user if they want to save the topic frequencies
if save_files == 'y':
    topic_counts_df = topic_counts.reset_index()
    topic_counts_df.columns = ['topic', 'count']
    topic_counts_df.to_csv(f"metadata/topic_frequencies_{filename}.csv", index=False, encoding="utf-8")

print(topic_counts, "\n")

# Frequency counts of 'organization'
print("Organization frequencies:")
organization_counts = df['organization'].value_counts()

# Save organization counts into file
if save_files == 'y':
    organization_counts_df = organization_counts.reset_index()
    organization_counts_df.columns = ['organization', 'count']
    organization_counts_df.to_csv(f"metadata/organization_frequencies_{filename}.csv", index=False, encoding="utf-8")
    
print(organization_counts, "\n")

#Frequency counts of 'fine_topic'
print("Fine topic frequencies:")
fine_topic_counts = df['fine_topic'].value_counts()

# Save fine topic counts into file
if save_files == 'y':
    fine_topic_counts_df = fine_topic_counts.reset_index()
    fine_topic_counts_df.columns = ['fine_topic', 'count']
    fine_topic_counts_df.to_csv(f"metadata/fine_topic_frequencies_{filename}.csv", index=False, encoding="utf-8")
    
print(fine_topic_counts, "\n")

# Relative frequencies (percentages)
print("Topic relative frequencies (%):")
print((topic_counts / len(df) * 100).round(2), "\n")

# Some basic stats on 'date'
print("Date stats:")
print(df['date'].describe(), "\n")

# Optionally, group by date to see number of entries per month
df['year'] = df['date'].str[:2]  # assuming date is YYMM
df['month'] = df['date'].str[2:4]
entries_per_month = df.groupby(['year', 'month']).size()
print("Entries per year/month:")
print(entries_per_month, "\n")

# Random sample of 5 headlines
print("Random 5 headlines:")
print(df['headline'].sample(5).values)
