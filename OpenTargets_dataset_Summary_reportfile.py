import re
import sys
import mygene
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
import json
import os
from tqdm import tqdm
from rpy2.robjects import default_converter, globalenv, FactorVector
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import umap.umap_ as umap
from adjustText import adjust_text

mg = mygene.MyGeneInfo()
tqdm.pandas()

limma = importr("limma")
base = importr("base")

# set if differential expression or not! [1 or 0]
# 0 for baseline and 1 for differential analysis
diffExp = 1
# set testing or not! [1 or 0]
# 0 for full run, 1 for testing (first 50 entries)
test = 0

def checktype():
    if diffExp == 1:
        print("Performing differential expression analysis.")
    elif diffExp == 0:
        print("Performing baseline analysis.")
    else:
        print("Mention baseline or differential expression analysis\n"
              "diffExp = 1 for differential expression analysis\n"
              "diffExp = 0 for baseline expression analysis")
        sys.exit("Exiting: 'diffExp' not set.")

def testing():
    if test == 1:
        print("Performing Test run using 50 sample observations.")
    elif test == 0:
        print("Performing Full run using complete dataset.")
    else:
        print("Mention if Test run or Full analysis\n"
              "test = 1 for a test run using first 50 entries\n"
              "test = 0 for a full run.")
        sys.exit("Exiting: 'test' not set.")

checktype()
testing()


path = "/Users/ananth/Documents/OpenTargets/PXD010138/OPTAR/"
# 1. Sample Metadata
SDRF = pd.read_csv(os.path.join(path, "PXD010138.sdrf.tsv"), sep='\t', header=0)

samples = (SDRF['source name'].unique().tolist())
dataset = SDRF['comment[proteomexchange accession number]'].unique()[0]
dataset_URL = SDRF['comment[file uri]'].str.replace(r'/[^/]+$', '', regex=True).unique()[0]

species = SDRF['characteristics[organism]'].unique().tolist()
speciesOntURI = "http://purl.obolibrary.org/obo/NCBITaxon_9606"
pubmedId = "31207390"
provider = "Mendonça CF, Kuras M. etal."
emailID = "gyorgy.marko-varga@bme.lth.se"
experimentType = "Proteomics by mass spectrometry"
quantificationMethod = "Label-free (differential)"
searchDatabase = "Human 'one protein per gene set' proteome (UniProt, November 2024. 20,656 sequences)"
contaminantDatabase = "cRAP contaminants (May 2021. 245 sequences)"
entrapmentDatabase = "Generated using method described by Wen B. etal. (PMID:40524023, 20,653 sequences)"
analysisSoftware = "MaxQuant v2.7.0.0"
operatingSystem = "Red Hat Enterprise Linux Server"

SDRF['experimentId'] = dataset
SDRF['experimentType'] = experimentType
SDRF['species'] = species[0]
SDRF['speciesOntURI'] = speciesOntURI
SDRF['pubmedIds'] = pubmedId
SDRF['provider'] = provider

SDRF.rename(columns={'source name': 'assayId',
                     'assay name': 'assayGroup',
                     'characteristics[organism part].1': 'tissue',
                     'characteristics[disease]': 'disease',
                     'characteristics[biological replicate]': 'individual',
                     'characteristics[sex]': 'sex',
                     'characteristics[age]': 'age'}, inplace=True)

sub_SDRF = SDRF[["assayId", "assayGroup", "tissue",
                 "disease", "individual", "sex", "age",
                 "experimentId", "experimentType", "species", "speciesOntURI",
                 "pubmedIds", "provider"]].drop_duplicates()

# convert dictionary to json
sdrf_json = (sub_SDRF.groupby(['experimentId', 'experimentType', 'species',
                               'speciesOntURI', 'pubmedIds', 'provider'])
             [['assayId', 'assayGroup', 'tissue', 'disease', 'individual', 'sex', 'age']]
             .apply(lambda x: x.to_dict('records'))
             .reset_index(name='experimentalDesigns')
             .to_dict('records'))

optar_result_dir = os.path.join(path, "MaxQuant/")
os.makedirs(optar_result_dir, exist_ok=True)

# write to a file
with open(os.path.join(optar_result_dir, dataset + "_OpenTargets_sdrf.json"), 'w') as outfile:
    json.dump(sdrf_json, outfile, indent=4)

# Create Metadata table
metatable = {
    'Detail': [dataset, dataset_URL, provider, emailID, pubmedId,
               experimentType, quantificationMethod,
               searchDatabase, contaminantDatabase, entrapmentDatabase,
               analysisSoftware, operatingSystem]
}
metarownames = ['Dataset identifier:', 'Dataset URL:', 'Dataset submitters:', 'E-mail:', 'PubMed ID:',
                'Experiment type:', 'Quantification method:', 'Search database:', 'Contaminant database:',
                'Entrapment database:', 'Analysis software:', 'Operating system:']

metadatatable = pd.DataFrame(metatable, index=metarownames)

metatab, ax_meta = plt.subplots(figsize=(8, 6))
ax_meta.axis('off')
ax_meta.table(cellText=metadatatable.values,
              colLabels=metadatatable.columns,
              rowLabels=metadatatable.index,
              loc='center',
              cellLoc='left')
metatab.text(0.02, 0.95, 'Summary of reanalysed PRIDE Mass Spectrometry proteomics dataset exported into Open Targets',
             ha='left', va='top', wrap=True, fontsize=12)

# 2. Protein Quantification data
ProteinGroups = pd.read_csv(os.path.join(path, "proteinGroups.txt"), sep='\t', header=0)
ProteinGroups = ProteinGroups.loc[:, ~ProteinGroups.columns.str.contains("Pool|Exclude|GIS", regex=True)]
# Pre-processed
contams = ProteinGroups[ProteinGroups['Potential contaminant'].str.contains(r'\+', regex=True, na=False)
                        | ProteinGroups['Protein IDs'].str.contains('CON_')
                        | ProteinGroups['Protein IDs'].str.contains('CONTAM_')]
entraps = ProteinGroups[ProteinGroups['Protein IDs'].str.contains('ENTRAP_')]
revdeco = ProteinGroups[ProteinGroups['Reverse'].str.contains(r'\+', regex=True, na=False)
                        | ProteinGroups['Protein IDs'].str.contains('REV_')
                        | ProteinGroups['Protein IDs'].str.contains('DECOY_')]

exclude = pd.concat([contams['Protein IDs'], revdeco['Protein IDs'], entraps['Protein IDs']])

# Remove decoys, contams and entraps before counting number of proteins and peptides identified in preprocessed data
ProteinGroups = ProteinGroups[~ProteinGroups['Protein IDs'].isin(exclude)]

# Count number of unique proteins
# Step 1: split on ';' on Protein IDs column
split_IDs = ProteinGroups['Protein IDs'].dropna().str.split(';')
# Step 2: Flatten the list of lists
all_PIDs = [item for sublist in split_IDs for item in sublist]
# Step 3: Count unique Protein IDs
Preprocessed_num_of_identified_proteins = pd.Series(all_PIDs).nunique()

# Total number of peptides
Preprocessed_num_of_mapped_peptides = ProteinGroups['Peptides'].sum()
# Total number of unique peptides
Preprocessed_num_of_unique_peptides = ProteinGroups['Unique peptides'].sum()

# keep protein groups identified by more than 1 unique peptide
ProteinGroups = ProteinGroups[ProteinGroups['Unique peptides'] > 1]

# Post-processing
# Fraction Of Total (FOT) normalisation
Postprocessed = ProteinGroups.copy()

# If TMT dataset
label = SDRF['comment[label]'].str.contains('TMT', case=False, na=False)
if label.any():
    internal_standard_labels = SDRF.loc[
        SDRF['disease'].str.lower().isin(['global internal standard', 'gis', 'pool']),'assayGroup'].unique().tolist()
    internal_standard_labels = ['Reporter intensity ' + x for x in internal_standard_labels]
    # remove intensities of Internal Standard TMT channels from downstream postprocessing
    Postprocessed = Postprocessed.drop(columns=internal_standard_labels)
    intensity_cols = Postprocessed.columns[
        Postprocessed.columns.str.contains("Reporter intensity \d+", regex=True)
    ].tolist()

else:
    # To get all iBAQ columns from proteinGroups.txt
    intensity_cols_pgroups = Postprocessed.columns[
        Postprocessed.columns.str.contains("^iBAQ", regex=True) &
        ~Postprocessed.columns.str.contains("iBAQ peptides", regex=True) &
        (Postprocessed.columns != "iBAQ")
    ].tolist()
    # To get only those iBAQ columns that are mentioned in SDRF from proteinGroups.
    intensity_cols = SDRF['assayGroup'].str.replace(r'^', 'iBAQ ', regex=True).unique().tolist()

    #####
    # To CHECK: if iBAQ sample names in SDRF are same as in proteinGroups.txt
    missing_samples = set(intensity_cols) - set(Postprocessed.columns)
    if missing_samples:
        print("Error: sample names in SDRF[assay name] are not the same as those in ProteinGroups.txt.")
        print("Note: SDRF[assay name] should not contain \"iBAQ\" just sample names as it appears in ProteinGroups.txt")
        print(missing_samples)
        sys.exit(1)
    #####
    # To CHECK: if iBAQ samples in proteinGroups are missing in SDRF?
    missing_in_sdrf = [col for col in intensity_cols_pgroups if col not in intensity_cols]
    if missing_in_sdrf:
        print("Warning: These iBAQ samples in ProteinGroups.txt are missing from SDRF.\n"
            "Only samples mentioned in SDRF[sample name] will be processed.\n")
        print(missing_in_sdrf)

# Fraction Of Total: Divide abundance of each protein by the total abundance of its sample (column) and scale it up
# to a billion to arrive at normalised abundance of parts per billion.
Postprocessed[intensity_cols] = Postprocessed[intensity_cols].div(Postprocessed[intensity_cols].sum(axis=0), axis=1) * 1000000000

# For testing
if test == 1:
    Postprocessed = Postprocessed.head(50)

# Map UniProt protein IDs to Ensembl Gene IDs
def map_GeneID(proteingroup):
    id = proteingroup['Protein IDs']
    split_IDs = id.split(';')
    #all_PIDs = [PID.split('|')[1] for PID in split_IDs]
    all_PIDs = [PID.split('|')[1] if '|' in PID else PID for PID in split_IDs]

    tmp = mg.querymany(all_PIDs, scopes="uniprot", fields='ensembl.gene,symbol', species='human', as_dataframe=True)

    if 'notfound' in tmp:
        tmp_ENSG = pd.NA
        tmp_Symb = pd.NA
    else:
        if 'ensembl.gene' in tmp.columns:
            tmp_ENSG = ";".join(tmp['ensembl.gene'].dropna().unique().astype(str))
        else:
            tmp_ENSG = pd.NA

        if 'symbol' in tmp.columns:
            tmp_Symb = tmp['symbol'].dropna().unique().tolist()
            tmp_Symb = ';'.join(tmp_Symb)
        else:
            tmp_Symb = pd.NA

    res = "#".join([str(tmp_ENSG), str(tmp_Symb)])
    return res


Postprocessed['genemap'] = Postprocessed.progress_apply(map_GeneID, axis=1)
Postprocessed['ENSG'] = Postprocessed['genemap'].str.split('#').str[0]
Postprocessed['Gene Symbol'] = Postprocessed['genemap'].str.split('#').str[1]
Postprocessed['Gene Count'] = Postprocessed['Gene Symbol']. \
    apply(lambda x: sum(1 for item in x.split(";") if item != "<NA>" and item != ''))

# Remove protein groups mapped to more than one Gene ID
Postprocessed = Postprocessed[Postprocessed['Gene Count'] == 1]
Postprocessed = Postprocessed.drop('genemap', axis=1)

# Count number of peptides and proteins after post-processed filtering
Postprocessed_num_of_mapped_peptides = Postprocessed['Peptides'].sum()
Postprocessed_num_of_unique_peptides = Postprocessed['Unique peptides'].sum()

post_split_IDs = Postprocessed['Protein IDs'].dropna().str.split(';')
post_allPIDs = [item for sublist in post_split_IDs for item in sublist]
Postprocessed_num_of_identified_proteins = pd.Series(post_allPIDs).nunique()

selected = ['ENSG', 'Gene Symbol', 'Protein IDs'] + intensity_cols

Postprocessed_iBAQ = Postprocessed[selected].copy()
Postprocessed_iBAQ = Postprocessed_iBAQ.replace(0, np.nan)

# Remove term iBAQ from column names
Postprocessed_iBAQ.columns = Postprocessed_iBAQ.columns.str.replace(r'^iBAQ |^Reporter intensity ', '', regex=True)

# Replace descriptive column names (Heart 1) with Source names (PXD-Sample-1)
unique_sample_names = SDRF[['assayGroup', 'assayId', 'disease']].drop_duplicates()

rename_dict = dict(zip(unique_sample_names['assayGroup'], unique_sample_names['assayId']))

Postprocessed_iBAQ = Postprocessed_iBAQ.rename(columns=rename_dict)
source_names = Postprocessed_iBAQ.columns[3:].tolist()

# Sort alphabetically by Gene names
Postprocessed_iBAQ = Postprocessed_iBAQ.sort_values(by='Gene Symbol')

######################
# Sample name to source name map table for report summary document
if label.any():
    # For TMT channels mention disease
    if "factor value[disease]" in SDRF.columns:
        mapping_table = SDRF[['factor value[disease]', 'assayId']].drop_duplicates()
        mapping_table.rename(columns={'assayId': 'sample name',
                                      'factor value[disease]': 'assay name'}, inplace=True)
    elif "factor value[organism part]" in SDRF.colums:
        mapping_table = SDRF[['factor value[organism part]', 'assayId']].drop_duplicates()
        mapping_table.rename(columns={'assayId': 'sample name',
                                      'factor value[organism part]': 'assay name'}, inplace=True)
    else:
        mapping_table = SDRF[['disease', 'assayId']].drop_duplicates()
        mapping_table.rename(columns={'assayId': 'sample name',
                                      'disease': 'assay name'}, inplace=True)

    mapping_table = mapping_table[mapping_table['sample name'].isin(source_names)]
    mapping_table = mapping_table.sort_values(by='sample name')

else:
    mapping_table = unique_sample_names[unique_sample_names['assayId'].isin(source_names)]
    mapping_table = mapping_table.sort_values(by='assayId')
    mapping_table.rename(columns={'assayId': 'sample name',
                                  'assayGroup': 'assay name'}, inplace=True)

maptab, ax6 = plt.subplots(figsize=(8, 10))
ax6.axis('off')
ax6.table(cellText=mapping_table.values,
          colLabels=mapping_table.columns,
          loc='center',
          cellLoc='left')
# maptable_caption = "Description of source and sample names"
# maptab.text(0.5, 0.02, maptable_caption, wrap=True, ha='center', va='bottom', fontsize=10)
# plt.tight_layout()

######################
# Create summary table
table = {
    'Pre-processed': [len(source_names), len(contams), len(entraps), len(revdeco),
                      Preprocessed_num_of_identified_proteins, Preprocessed_num_of_mapped_peptides,
                      "NA", Preprocessed_num_of_unique_peptides],
    'Post-processed*': [len(source_names), 0, 0, 0,
                        Postprocessed_num_of_identified_proteins, Postprocessed_num_of_mapped_peptides,
                        len(Postprocessed), Postprocessed_num_of_unique_peptides]
}
rownames = ['Number of samples', 'Number of potential contaminants˚', 'Number of entrapments¶',
            'Number of reverse decoys^',
            'Number of identified proteins†', 'Total number of mapped peptides¥',
            'Protein groups mapped to unique gene id¢', ' |->.....Number of mapped unique peptides§']

datatable = pd.DataFrame(table, index=rownames)

print(datatable)

tab, ax0 = plt.subplots(figsize=(8, 6))
ax0.axis('off')

ax0.table(cellText=datatable.values,
          colLabels=datatable.columns,
          rowLabels=datatable.index,
          loc='center',
          cellLoc='left')

tab.text(0.02, 0.95,
         'Output before and after processing.\n\n\
The submitted original ‘.raw’ files are run through MaxQuant; \
the output (pre-processed) intensities are then FOT normalised \
proteins mapped to Ensembl gene IDs and filtered results \
(post-processed) are uploaded to Open Targets',
         ha='left', va='top', wrap=True, fontsize=12)
plt.tight_layout()

Table_caption = "* Data show in Open Targets.\n\
˚ The total number of protein groups found to be a commonly occurring contaminant.\n\
¶ Total number of entrapment proteins.\n\
ˆ The total number of protein groups with a peptide derived from the reversed part of the decoy database.\n\
† The total number of non-isoform SwissProt proteins within the protein group, to which at least 2 or more peptides from each sample are mapped to.\n\
¥ Sum of peptides that are mapped across all protein groups.\n\
¢ The total number of protein groups which are mapped to an unique Ensembl Gene ID.\n\
§ The total number of unique peptides associated with the protein group (i.e. these peptides are not shared with another protein group)."

tab.text(0.02, 0.02, Table_caption, wrap=False, ha='left', va='bottom', fontsize=10)
plt.tight_layout()

##############################################
# Figure 1. Postprocessed iBAQ values across samples
##############################################
Postprocessed_iBAQ_long = pd.melt(Postprocessed_iBAQ,
                                  id_vars=['ENSG', 'Gene Symbol', 'Protein IDs'],
                                  value_vars=source_names,
                                  var_name='Sample',
                                  value_name='FOT normalised iBAQ (ppb)')

sorted_samples = sorted(Postprocessed_iBAQ_long['Sample'].unique())

if Postprocessed_iBAQ.shape[1] > 40:
    plotheight = Postprocessed_iBAQ.shape[1] / 2.5
    fig1 = plt.figure(figsize=(10, plotheight))
    ax1 = fig1.add_subplot(111)
    sb.boxplot(y='Sample', x='FOT normalised iBAQ (ppb)', data=Postprocessed_iBAQ_long, order=sorted_samples, ax=ax1)
    ax1.set_xscale("log")
    ax1.set_title('FOT normalised iBAQ')

else:
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    sb.boxplot(x='Sample', y='FOT normalised iBAQ (ppb)', data=Postprocessed_iBAQ_long, order=sorted_samples, ax=ax1)
    ax1.tick_params(axis='x', rotation=90)
    ax1.set_yscale("log")
    ax1.set_title('FOT normalised iBAQ')

plt.tight_layout(rect=[0, 0.1, 1, 0.95])

# figure1 caption (text below plot)
Figure1_caption = "Figure 1: Boxplots with distribution of iBAQ values for each sample after FOT normalisation."
fig1.text(0.5, 0.02, Figure1_caption, wrap=True, horizontalalignment='center', fontsize=10)

# Write post-processed quant values as JSON
def convertToJSON(df):
    # Replace descriptive column names (Heart 1) with Source names (PXD-Sample-1)
    matrixlong = df.copy()
    matrixlong['Unit'] = 'ppb'
    matrixlong.columns = ['Gene ID', 'Gene Symbol', 'UniProt ID', 'assayID', 'value', 'Unit']

    postprocessed_json = (
        matrixlong.groupby(['Gene ID', 'Gene Symbol', 'UniProt ID', 'Unit'])[['assayID', 'value']]
        .apply(lambda x: x.to_dict('records'))
        .reset_index(name='Expression')
        .to_dict('records'))

    # Filter out Expression entries with value == 0.0 for each gene
    for gene in postprocessed_json:
        gene["Expression"] = [entry for entry in gene["Expression"] if entry["value"] != 0.0 and pd.notna(entry["value"])]

    # Sort alphabetically by "Gene Symbol"
    postprocessed_json = sorted(postprocessed_json, key=lambda x: x["Gene Symbol"])

    return(postprocessed_json)


if diffExp != 1:
    # Write Post-processed baseline abundance matrix
    with open(os.path.join(optar_result_dir, dataset + "_OpenTargets_ppb.txt"), 'w') as outfile:
        Postprocessed_iBAQ.to_csv(outfile, sep='\t', index=False)

    # Write Post-processed baseline results to json to a file
    postprocessed_json = convertToJSON(Postprocessed_iBAQ_long)

    with open(os.path.join(optar_result_dir, dataset + "_OpenTargets_ppb.json"), 'w') as outfile:
        json.dump(postprocessed_json, outfile, indent=4)

    print("Quant files written.")

##############################################
# Figure 2. Protein groups commonly identified across samples
# count number of proteins identified across all samples
##############################################
Postprocessed_iBAQ['protein_count_across_samples'] = (
        (Postprocessed_iBAQ.iloc[:, 3:] != 0) &
        (pd.notna(Postprocessed_iBAQ.iloc[:, 3:]))).sum(axis=1)
Proteins_identified_across_samples = Postprocessed_iBAQ[
    ['ENSG', 'Gene Symbol', 'Protein IDs', 'protein_count_across_samples']]
proteincounts_across_sample = Proteins_identified_across_samples[
    'protein_count_across_samples'].value_counts().sort_index()

if Postprocessed_iBAQ.shape[1] > 40:
    plotheight = Postprocessed_iBAQ.shape[1] / 2.5
    fig2 = plt.figure(figsize=(10, plotheight))
    ax2 = fig2.add_subplot(111)
    proteincounts_across_sample.plot(kind='barh', ax=ax2)
    ax2.set_ylabel('Number of samples')
    ax2.set_xlabel('Number of protein groups')
    ax2.set_title('Protein groups commonly identified across samples')
else:
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    proteincounts_across_sample.plot(kind='bar', ax=ax2)
    ax2.set_xlabel('Number of samples')
    ax2.set_ylabel('Number of protein groups')
    ax2.set_title('Protein groups commonly identified across samples')

Figure2_caption = "Figure 2: Indicates the number of protein groups that were identified across different samples.\
Protein groups were counted as present in a sample when the sample had registered intensity."
fig2.text(0.5, -0.05, Figure2_caption, wrap=True, horizontalalignment='center', fontsize=10)

##############################################
# Figure 3. Total number of proteins identified in each sample
##############################################
tmp = Postprocessed_iBAQ_long[['Sample', 'FOT normalised iBAQ (ppb)']].copy()
tmp = tmp[tmp['FOT normalised iBAQ (ppb)'] > 0]
proteincounts_each_sample = tmp['Sample'].value_counts().sort_index()

if Postprocessed_iBAQ.shape[1] > 40:
    plotheight = Postprocessed_iBAQ.shape[1] / 2.5
    fig3 = plt.figure(figsize=(10, plotheight))
    ax3 = fig3.add_subplot(111)
    proteincounts_each_sample.plot(kind='barh', ax=ax3)
    ax3.set_title('Protein groups identified in each sample')
    ax3.set_ylabel('Samples')
    ax3.set_xlabel('Number of protein groups')
    ax3.tick_params(axis='x', rotation=90)
else:
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    proteincounts_each_sample.plot(kind='bar', ax=ax3)
    ax3.set_title('Protein groups identified in each sample')
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Number of protein groups')
    ax3.tick_params(axis='x', rotation=90)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])

Figure3_caption = "Figure 3: Protein counts in each sample. The total number of proteins (SwissProt non-isoforms) \
from all protein groups to which at least 2 or more unique peptides from each sample are mapped to."
fig3.text(0.5, 0.02, Figure3_caption, wrap=True, horizontalalignment='center', fontsize=10)

#
##############################################
# Figure 4. Total number of peptides mapped per sample
# NOTE: Individual sample level peptide count data is not available
# for individual TMT channels in proteinGroups.txt,
# hence this plot is only shown for label-free experiments.
##############################################
if not label.any():
    peptide_cols = SDRF['assayGroup'].str.replace(r'^', 'Peptides ', regex=True).unique().tolist()
    Sample_peptides = Postprocessed[peptide_cols].copy()
    Sample_peptides['ENSG'] = Postprocessed['ENSG']

    Sample_peptides.columns = Sample_peptides.columns.str.replace(r'^Peptides ', '', regex=True)
    Sample_peptides = Sample_peptides.rename(columns=rename_dict)

    Sample_peptides_long = pd.melt(Sample_peptides,
                                   id_vars=['ENSG'],
                                   value_vars=Sample_peptides.columns[:-1],
                                   var_name='Samples',
                                   value_name='Number of peptides'
                                   )
    Sample_peptides_long = Sample_peptides_long[Sample_peptides_long['Number of peptides'] > 0]
    peptidecounts_each_sample = Sample_peptides_long['Samples'].value_counts().sort_index()

    if Postprocessed_iBAQ.shape[1] > 40:
        plotheight = Postprocessed_iBAQ.shape[1] / 2.5
        fig4 = plt.figure(figsize=(10, plotheight))
        ax4 = fig4.add_subplot(111)
        peptidecounts_each_sample.plot(kind='barh', ax=ax4)
        ax4.set_title('Peptides identified in each sample')
        ax4.set_ylabel('Samples')
        ax4.set_xlabel('Number of peptides')
        ax4.tick_params(axis='x', rotation=90)
    else:
        fig4 = plt.figure(figsize=(10, 6))
        ax4 = fig4.add_subplot(111)
        peptidecounts_each_sample.plot(kind='bar', ax=ax4)
        ax4.set_title('Peptides identified in each sample')
        ax4.set_xlabel('Samples')
        ax4.set_ylabel('Number of peptides')
        ax4.tick_params(axis='x', rotation=90)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    Figure4_caption = "Figure 4: Peptide counts in each sample. The total number of peptides that are mapped\
    across all protein groups from each sample."
    fig4.text(0.5, 0.02, Figure4_caption, wrap=True, horizontalalignment='center', fontsize=10)

##############################################
# Figure 5. Corrleation Heatmap
##############################################
tmp = Postprocessed_iBAQ[source_names].copy()
corr_matrix = tmp.corr(numeric_only=True)
# make sure corr_matrix has no NaN values
corr_matrix = corr_matrix.fillna(0)
# print(np.isfinite(corr_matrix.values).all())
# print(np.where(~np.isfinite(corr_matrix.values)))

if Postprocessed_iBAQ.shape[1] > 40:
    #plotheight = Postprocessed_iBAQ.shape[1] / 2.5
    plotheight = 20
else:
    plotheight = 10

# Plot using clustermap
fig5 = sb.clustermap(corr_matrix,
                     cmap="YlGnBu",
                     figsize=(plotheight, plotheight),
                     annot=False,
                     dendrogram_ratio=(0.05, 0.15),
                     cbar_pos=(0.02, 0.8, 0.02, 0.15),
                     row_cluster=False,
                     col_cluster=True)

fig5.fig.suptitle("Clustered map of Perasons correlation between samples", y=1.02, fontsize=14)
plt.subplots_adjust(bottom=0.2)
plt.figtext(0.5, 0.05,
            "Figure 5: Clustered map of correlation between samples. "
            "The pairwise Pearson correlation was calculated between normalised intensities (iBAQs) of each sample and clustered hierarchically.",
            wrap=True, ha='center', fontsize=10)

glossary = """
Post-processing filters applied:
 (i) Remove reverse decoys.
 (ii) Remove potential contaminants.
 (iii) Include protein groups to which 2 or more unique peptides are mapped.
 (iv) Include protein groups wherein all protein IDs within are mapped to an unique Ensembl Gene ID.

Normalisation method:
 Fraction Of Total (FOT): Each protein iBAQ intensity value is scaled to the total amount of signal in a given MS run (column) and transformed to parts per billion (ppb)

Reverse decoy: This particular protein group contains no protein, made up of at least 50% of the peptides of the leading protein, with a peptide derived from the reversed part of the decoy database. These are removed for further data analysis. The 50% rule is in place to prevent spurious protein hits to erroneously flag the protein group as reverse.

Potential contaminant: This particular protein group was found to be a commonly occurring contaminant. These are removed for further data analysis.

Entrapments: Procedures to generate entrapment peptide sequence database were generated using the methods described in Wen B. etal. (PMID:40524023)

Peptides: The total number of peptide sequences associated with the protein group (ie. for all the proteins in the group).

Unique peptides: The total number of unique peptides associated with the protein group (ie. these peptides are not shared with another protein group).
"""
glos_text = plt.figure(figsize=(8, 6))
ax6 = glos_text.add_subplot(111)
plt.text(0.01, 0.99, glossary, ha='left', va='top', wrap=True, fontsize=12)
ax6.set_title('Glossary', loc='left')
plt.axis('off')

##########################################
# Differential Expression Analysis section

#############################
# Limma Batch effect correction
def limma_batchEffect(ibaq_matrix, batch_annotation):

    matrix_df = ibaq_matrix.copy()
    batch = batch_annotation.copy()

    # Activate the R-Python interface
    # Load R libraries
    ro.r('library(limma)')

    # Convert and assign ibaq_matrix to R
    with localconverter(default_converter + pandas2ri.converter):
        globalenv['expr'] = ro.conversion.py2rpy(matrix_df)

    # Convert and assign batch to R as a factor
    globalenv['batch'] = FactorVector(batch)

    # Run removeBatchEffect
    print("Performing Limma batch correction")
    ro.r('expr_corrected <- removeBatchEffect(expr, batch=batch)')

    # Get the result from R back to Python
    with localconverter(default_converter + pandas2ri.converter):
        expr_limma_corrected = ro.conversion.rpy2py(ro.r['expr_corrected'])

    # Assign row and column names
    expr_limma_corrected = pd.DataFrame(expr_limma_corrected,
                                        columns=ibaq_matrix.columns,
                                        index=ibaq_matrix.index)

    return expr_limma_corrected


if diffExp == 1:
    # read batch annotation file
    batch_annotation = pd.read_csv(os.path.join(path, "Limma_annotation.txt"), sep='\t', header=0)

    ibaq_matrix = Postprocessed_iBAQ.copy()

    ibaq_matrix = ibaq_matrix[source_names].set_index(
        ibaq_matrix[['ENSG', 'Gene Symbol', 'Protein IDs']].agg('+'.join, axis=1))

    ibaq_matrix = ibaq_matrix.replace('nan', np.nan)
    # IMPORTANT: Log transform iBAQ values before Batch Effect correction and Differential Expression.
    ibaq_matrix = np.log2(ibaq_matrix + 1)

    # Check: find if any samples are missing in the annotation
    missing_samples = set(ibaq_matrix.columns) - set(batch_annotation["Sample name"])
    if missing_samples:
        raise ValueError(f"Samples in ibaq_matrix not found in batch annotation file: {missing_samples}\n")

    colnames = ibaq_matrix.columns.tolist()
    matrix_colnames = pd.DataFrame(colnames, columns=["Sample name"])
    batch_annotation = pd.merge(matrix_colnames, batch_annotation, on="Sample name")
    batch_annotation = batch_annotation.sort_values(by="Sample name")
    batch = batch_annotation['Batch'].tolist()

    # IMPORTANT: Sort and arrange iBAQ matrix columns to match and be in the same order as batch_annotation
    ibaq_matrix = ibaq_matrix[batch_annotation["Sample name"].values]

    num_of_batches = batch_annotation['Batch'].nunique()

    # Perform limma batch effect correction only if there are more than 1 batch.
    if num_of_batches > 1:
        expr_limma_corrected = limma_batchEffect(ibaq_matrix, batch)
    else:
        expr_limma_corrected = ibaq_matrix.copy()

    # Reconvert log2 transformed limma corrected values back to save as iBAQ ppb values and Write abundance matrix
    tmp_exp = expr_limma_corrected.copy()
    expr_limma_iBAQ = (2 ** tmp_exp) - 1

    expr_limma_iBAQ['ENSG'] = expr_limma_iBAQ.index.str.replace(r'\+.*', '', regex=True)
    expr_limma_iBAQ['Gene Symbol'] = expr_limma_iBAQ.index.str.replace(r'ENSG\d+\+', '', regex=True)
    expr_limma_iBAQ['Gene Symbol'] = expr_limma_iBAQ['Gene Symbol'].str.replace(r'\+.*', '', regex=True)
    expr_limma_iBAQ['Protein IDs'] = expr_limma_iBAQ.index.str.replace(r'.*\+', '', regex=True)
    cols = expr_limma_iBAQ.columns.tolist()
    column_reorder = cols[-3:] + cols[:-3]
    expr_limma_iBAQ = expr_limma_iBAQ[column_reorder]

    matrix_df_long = pd.melt(expr_limma_iBAQ,
                             id_vars=['ENSG', 'Gene Symbol', 'Protein IDs'],
                             value_vars=source_names,
                             var_name='Sample',
                             value_name='FOT normalised iBAQ (ppb)')

    # Write Post-processed results matrix
    with open(os.path.join(optar_result_dir, dataset + "_OpenTargets_ppb.txt"), 'w')as outfile:
        expr_limma_iBAQ.to_csv(outfile, sep='\t', index=False)

    # Write Post-processed results to json to a file
    postprocessed_json = convertToJSON(matrix_df_long)
    with open(os.path.join(optar_result_dir, dataset + "_OpenTargets_ppb.json"), 'w') as outfile:
        json.dump(postprocessed_json, outfile, indent=4)

    print("Quant files written.")

    # Save batch annotation table to summary pdf
    batch_annot_tab, ax9 = plt.subplots(figsize=(8, 10))
    ax9.axis('off')
    ax9.table(cellText=batch_annotation[['Sample name','Condition','Batch']].values,
              colLabels=batch_annotation[['Sample name','Condition','Batch']].columns,
              loc='center',
              cellLoc='left',
              fontsize=12)
    #plt.tight_layout()

    ## UMAP
    print("Performing UMAP.")
    # transpose to have rows as samples and columns as features (genes, peptides, etc.).
    expr_limma_trans = expr_limma_corrected.copy().T
    # change NaN to 0. UMAP does not handle NaN
    expr_limma_trans[np.isnan(expr_limma_trans)] = 0
    # Initialize UMAP, Fit and transform
    umap_plotdata = umap.UMAP(n_components=2, random_state=42).fit_transform(expr_limma_trans)

    umap_plotdata = pd.DataFrame(umap_plotdata,
                                 columns=["UMAP1", "UMAP2"],
                                 index=expr_limma_corrected.columns.tolist())

    umap_plotdata.index.name = 'assayId'
    umap_plotdata = umap_plotdata.reset_index()
    umap_plotdata = pd.merge(umap_plotdata, unique_sample_names, on='assayId')
    umap_plotdata['Batch'] = batch
    umap_plotdata['Sample'] = umap_plotdata['assayGroup'].str.replace(r'\d+', '', regex=True).str.strip()
    umap_plotdata['Sample'] = umap_plotdata['Sample'].str.replace(r'_', ' ', regex=True).str.strip()
    umap_plotdata['Sample'] = umap_plotdata['Sample'].str.replace(r'(?i)Asymptomatic', 'Asym', regex=True)
    umap_plotdata['Sample'] = umap_plotdata['Sample'].str.replace(r'(?i)Alzheimer\'s disease', 'AD', regex=True)

    fig6 = plt.figure(figsize=(7, 5))
    ax7 = fig6.add_subplot(111)

    if label.any():
        sb.scatterplot(data=umap_plotdata, x="UMAP1", y="UMAP2", hue="disease", style="Batch")
    else:
        sb.scatterplot(data=umap_plotdata, x="UMAP1", y="UMAP2", hue="Sample", style="Batch")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax7.set_title('UMAP')
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    if num_of_batches > 1:
        Figure6_caption = "Figure 6: UMAP of limma corrected expression."
    else:
        Figure6_caption = "Figure 6: UMAP of iBAQ expression."

    fig6.text(0.5, 0.02, Figure6_caption, wrap=True, horizontalalignment='center', fontsize=10)

    '''
    ## PCA
    pca = PCA(n_components=2)
    pca_df = pca.fit_transform(expr_limma_trans)
    pca_df = pd.DataFrame(data=pca_df,
                          columns=['PC1', 'PC2'],
                          index=expr_limma_trans.index.tolist())
    pca_df['assayId'] = pca_df.index.tolist()
    pca_df = pd.merge(pca_df, unique_sample_names, on='assayId')
    pca_df['Sample'] = pca_df['assayGroup'].str.replace(r'\d+', '', regex=True).str.strip()
    pca_df['Sample'] = pca_df['Sample'].str.replace(r'(?i)Asymptomatic', 'Asym', regex=True)
    pca_df['Sample'] = pca_df['Sample'].str.replace(r'(?i)Alzheimer\'s disease', 'AD', regex=True)
    pca_df['Batch'] = batch

    sb.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Sample", style="Batch")
    plt.title("PCA")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    #plt.show()
    '''

    ## Differential expression
    print("Performing differential expression analysis.")

    # Arrange expr_limma_corrected matrix columns in the same order as batch annotation
    expr_limma_corrected = expr_limma_corrected[batch_annotation["Sample name"].values]

    # no spaces or special characters (except _ ) for conditions, limma does not allow special characters
    batch_annotation['Condition'] = batch_annotation['Condition'].str.replace(r'\s+|\'|,', '', regex=True).str.strip()
    group = batch_annotation['Condition'].tolist()

    with localconverter(default_converter + pandas2ri.converter):
        globalenv['limma_expr_df'] = ro.conversion.py2rpy(expr_limma_corrected)
    globalenv['group'] = ro.FactorVector(group)

    # Code in R to calculate differential log Fold Change
    ro.r('''
    design <- model.matrix(~ 0 + group)
    colnames(design) <- levels(group)
    group_levels <- levels(group)
    contrast_pairs <- combn(group_levels, 2, simplify = FALSE)

    fit <- lmFit(limma_expr_df, design)
    diff_result_all <- list()
    
    for (pair in contrast_pairs) {
        contrast_name <- paste(pair[1], "-", pair[2], sep="")
        contrast.matrix <- makeContrasts(contrast_name, levels=design)
        fit2 <- contrasts.fit(fit, contrast.matrix)
        fit2 <- eBayes(fit2)
        diff_result_all[[contrast_name]] <- topTable(fit2, number=Inf)
        #print(head(diff_result_all[[contrast_name]]))
    }

    ''')

    contrast_names = list(ro.r('names(diff_result_all)'))
    contrast_names = [str(name) for name in contrast_names]
    print(contrast_names)

    volcano_plots = {}
    def volcanoPlot(name, df):
        mat = df.copy()
        fc_threshold = 1
        #fc_cap = 4 #limit of fold change to show on volcano plot, so to avoid showing extreme outliers.
        pval_threshold = 0.05

        mat['neg_log10_p'] = -np.log10(mat['P.Value'])
        mat['Significance'] = 'Not Significant'
        mat.loc[(mat['P.Value'] < pval_threshold) & (mat['logFC'] > fc_threshold), 'Significance'] = 'Upregulated'
        mat.loc[(mat['P.Value'] < pval_threshold) & (mat['logFC'] < -fc_threshold), 'Significance'] = 'Downregulated'

        #df = df[(df['logFC']) <= 4 & (df['logFC'] >= -4)]

        # Create figure object
        fig7, ax8 = plt.subplots(figsize=(8, 5))
        sb.scatterplot(data=mat, x='logFC', y='neg_log10_p', hue='Significance',
                       palette={'Not Significant': 'grey', 'Upregulated': 'red', 'Downregulated': 'blue'},
                       alpha=0.7, ax=ax8)
        labels = []
        signific = mat[mat['Significance'] != 'Not Significant'].nlargest(10, 'neg_log10_p')

        # Add labels
        for _, row in signific.iterrows():
            labels.append(ax8.text(row['logFC'], row['neg_log10_p'], row['Gene Symbol'], fontsize=8))

        # Adjust text to prevent overlap
        adjust_text(labels, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

        ax8.axhline(-np.log10(pval_threshold), linestyle='--', color='black', lw=1)
        ax8.axvline(-fc_threshold, linestyle='--', color='black', lw=1)
        ax8.axvline(fc_threshold, linestyle='--', color='black', lw=1)

        ax8.set_title(f'Differential Expression: {name}')
        ax8.set_xlabel('log2 Fold Change')
        ax8.set_ylabel('-log10 P-value')
        #ax8.set_xlim(-fc_cap, fc_cap)
        ax8.get_legend().remove()
        #ax8.legend(title='Expression')

        Volcanoplot_caption = "Figure: Volcano plot of differential protein abundances. \
The canonical gene identifiers are shown instead of UniProt protein identifiers."
        fig7.text(0.5, 0.02, Volcanoplot_caption, wrap=True, horizontalalignment='center', fontsize=10)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save the figure object
        volcano_plots[name] = fig7
        return volcano_plots


    diff_results_py = {}
    for name in contrast_names:
        with localconverter(default_converter + pandas2ri.converter):
            result = ro.r(f'diff_result_all[["{name}"]]')
            result['ENSG'] = result.index.str.replace(r'\+.*', '', regex=True)
            result['Gene Symbol'] = result.index.str.replace(r'ENSG\d+\+', '', regex=True)
            result['Gene Symbol'] = result['Gene Symbol'].str.replace(r'\+.*', '', regex=True)
            result['Protein IDs'] = result.index.str.replace(r'.*\+', '', regex=True)
            diff_results_py[name] = ro.conversion.rpy2py(result)
            #print("df_py_name:", diff_results_py[name].head())

            with open(os.path.join(optar_result_dir, f"{dataset}_OpenTargets_{name}_FC.txt"), 'w') as outfile:
                diff_results_py[name].to_csv(outfile, sep='\t', index=False)

            volcanoplots = volcanoPlot(name, diff_results_py[name])

            print(f"Fold change values written: {name}.")



##############################################
# Save plots to suammary pdf
with PdfPages(optar_result_dir + dataset + '_OpenTargets_Summary_report.pdf') as pdf:
    print("Saving summary report file")

    pdf.savefig(metatab, bbox_inches='tight')
    plt.close(metatab)

    pdf.savefig(tab, bbox_inches='tight')
    plt.close(tab)

    pdf.savefig(fig1, bbox_inches='tight')
    plt.close(fig1)

    pdf.savefig(fig2, bbox_inches='tight')
    plt.close(fig2)

    pdf.savefig(fig3, bbox_inches='tight')
    plt.close(fig3)

    if not label.any():
        pdf.savefig(fig4, bbox_inches='tight')
        plt.close(fig4)

    pdf.savefig(fig5.fig, bbox_inches='tight')

    if(diffExp == 1):

        pdf.savefig(fig6, bbox_inches='tight')
        plt.close(fig6)

        for name, fig7 in volcanoplots.items():
            pdf.savefig(fig7, bbox_inches='tight')
            plt.close(fig7)

    if(diffExp == 1):
        pdf.savefig(batch_annot_tab, bbox_inches='tight')
        plt.close(batch_annot_tab)
    else:
        pdf.savefig(maptab, bbox_inches='tight')
        plt.close(maptab)

    pdf.savefig(glos_text, bbox_inches='tight')
    plt.close(glos_text)

print("Job completed.")
print("End")








