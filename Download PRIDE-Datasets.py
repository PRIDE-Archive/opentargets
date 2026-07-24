import requests
import json
import time
import csv

# Base API URL
base_url = "https://www.ebi.ac.uk/pride/ws/archive/v3/search/projects"
params = {
    #"keyword": "DIA, SWATH, Data Independent Acquisition",
    "keyword": "Homo sapiens",
    "pageSize": 100,
    "page": 0,
    "sortDirection": "DESC",
    "sortFields": "submissionDate"
}

# To find total entries of datasets
response = requests.get(base_url, params=params)
response.raise_for_status()

# Response headers
headers = response.headers

# Total number of matching records
total_records = int(headers["total_records"])

page_size = params["pageSize"]
total_pages = (total_records + page_size - 1) // page_size  # Round up

all_results = []

print(f"Fetching {total_records} entries in {total_pages} pages...")

for page in range(total_pages):
    params["page"] = page
    print(f"Downloading page {page + 1} of {total_pages}...", end=" ")

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"\n Failed at page {page} with status code {response.status_code}")
        break

    data = response.json()


    if isinstance(data, list):
        all_results.extend(data)
        print(f" {len(data)} records")
    else:
        print("Unexpected response type:", type(data))

    time.sleep(0.2)  # Be nice to the server


# Open a TSV file for writing
with open("/Users/ananth/Documents/OpenTargets/PRIDE_datasets.tsv", "w", newline='', encoding='utf-8') as out_file:
    writer = csv.writer(out_file, delimiter='\t')

    # Write header
    header = ["accession", "title", "numberofProjectFiles", "publicationDate", "instruments",
              "quantificationMethods", "organisms", "organismsParts", "diseases", "keywords", "experimentType"]
    writer.writerow(header)

    # Iterate through each dataset entry
    for entry in all_results:
        count = len(entry["projectFileNames"])
        row = [
            entry.get("accession", ""),
            entry.get("title", ""),
            count,
            entry.get("publicationDate", ""),
            "; ".join(entry.get("instruments", [])),
            "; ".join(entry.get("quantificationMethods", [])),
            "; ".join(entry.get("organisms", [])),
            "; ".join(entry.get("organismsPart", [])),
            "; ".join(entry.get("diseases", [])),
            "; ".join(entry.get("highlights", {}).get("keywords", [])),
            "; ".join(entry.get("experimentTypes", [])),
        ]
        writer.writerow(row)

print("Done! Data written to outfile tsv'.")
