# Opentargets - OTAR3091

# Proteomics Data Processing Pipeline

This repository contains the workflow used for selecting, processing, and post-processing public proteomics datasets from the PRIDE database for integration into the Open Targets Platform.

---

## Step 1. Dataset Selection

The complete list of datasets in the PRIDE database was obtained using the PRIDE API.

**Run:**

```bash
Download_PRIDE_Datasets.py
```

The resulting TSV file was manually filtered according to the dataset selection criteria:

- https://github.com/PRIDE-Archive/opentargets/blob/main/guidelines/Dataset_Selection.md

---

## Step 2. SDRF (Sample and Data Relationship Format)

Annotate sample metadata according to the SDRF guidelines.

**SDRF specification**

- https://github.com/bigbio/proteomics-sample-metadata/blob/master/sdrf-proteomics/README.adoc

**Annotated SDRF files**

- https://github.com/PRIDE-Archive/opentargets/tree/main/sdrf

---

## Step 3. Process Datasets

### DDA (MaxQuant)

Process raw files using **MaxQuant**.

> **Important:** The **assay name** field in the SDRF file **must exactly match** the **Experiment** field in MaxQuant.

### DIA (DIA-NN)

Process raw files using **DIA-NN**.

---

## Step 4. Post-process Results

### DDA (MaxQuant)

Run:

```bash
OpenTargets_dataset_Summary_reportfile.py
```

**Required input files**

- `proteinGroups.txt`
- SDRF file

### DIA (DIA-NN)

Run:

```bash
OpenTargets_DIA_Summary_report.py
```

**Required input files**

- `report.tsv`
- SDRF file

---

## Summary of Post-processing Scripts

| Data Type | Script |
|-----------|--------|
| DDA / TMT / iTRAQ | `OpenTargets_dataset_Summary_reportfile.py` |
| DIA | `OpenTargets_DIA_Summary_report.py` |

---

## Processed Data

Raw and post-processed results are available from the PRIDE FTP server:

https://ftp.pride.ebi.ac.uk/pub/databases/pride/resources/proteomes/otargets/

