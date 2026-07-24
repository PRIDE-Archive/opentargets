# opentargets

Steps for Proteomics Data processing <br>
<br>

Step1. Dataset selection<br>
Complete list of datasets in the PRIDE database (https://www.ebi.ac.uk/pride/) were listed using the PRIDE API<br>
Run Download_PRIDE_Datasets.py<br>
The datasets in the resulting tsv were manually filtered using this dataset selection criteria:<br>
https://github.com/PRIDE-Archive/opentargets/blob/main/guidelines/Dataset_Selection.md<br>
<br>

Step2. SDRF (Sample to Data Relationship Format)<br>
Sample metadata annotation of shortlisted dataasets to follow SDRF guidelines - https://github.com/bigbio/proteomics-sample-metadata/blob/master/sdrf-proteomics/README.adoc<br>
Annotated sample metadata of processed files: https://github.com/PRIDE-Archive/opentargets/tree/main/sdrf<br>
<br>

Step3. Process datasets<br>
Process raw files in MaxQuant - <br>
"Experiment" field in MaxQuant should be same as "assay name" field in SDRF<br>
Process raw files in DIA-NN - <br>
<br>

Step4. Postprocess results<br>
For postprocessing DDA results from MaxQuant - use OpenTargets_dataset_Summary_reportfile.py <br>
required files - proteinGroups.txt; SDRF <br>

For postprocessing DIA results from DIA-NN - use OpenTargets_DIA_Summary_report.py <br>
required files - report.tsv; SDRF <br>

Summary of postprocess results<br>
For DDA/TMT/iTRAQ from MaxQuant use: OpenTargets_dataset_Summary_reportfile.py<br>
For DIA from DIA-NN use: OpenTargets_DIA_Summary_report.py<br>

Raw and Postprocess results on FTP<br>
https://ftp.pride.ebi.ac.uk/pub/databases/pride/resources/proteomes/otargets/<br>

