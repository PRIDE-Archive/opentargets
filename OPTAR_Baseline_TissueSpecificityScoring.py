import pyarrow
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from numpy.ma.core import shape
from rpy2.robjects import default_converter, globalenv, FactorVector
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import umap.umap_ as umap
import seaborn as sb
import plotly.express as px

path = "/Users/ananth/Documents/OpenTargets/"

###############################
# Read all Open Target baseline parquet files from directory
# and extract only proteomics entries
###############################
#table = pq.ParquetDataset(os.path.join(path, "Test_parquet_files")).read()
#df = table.to_pandas()
#sub = df[df['datatypeId'] == "mass-spectrometry proteomics"]
# write to a file
#with open(os.path.join(path + "OTAR_unaggregated_proteomics.txt"), 'w') as outfile:
#    sub.to_csv(outfile, sep='\t', index=False)

OTAR_alldata = pd.read_csv(os.path.join(path, "OPTAR_unaggregated_proteomics.txt"), sep='\t', header=0)

OTAR_alldata_clean = OTAR_alldata.drop_duplicates()

OTAR_alldata_clean['donorTissueId'] = OTAR_alldata_clean['donorId'] + "_" + OTAR_alldata_clean['tissueBiosampleFromSource']

OTAR_wide = (
    OTAR_alldata_clean.pivot_table(
        index = ['targetId', 'datasourceId', 'datatypeId', 'unit',
                 'targetFromSourceId'],
        #       'tissueBiosampleFromSource','tissueBiosampleId', 'age', 'sex',
        columns = 'donorTissueId',
        values = 'expression',
        aggfunc = 'mean').reset_index()
)

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

    # IMPORTANT: Important that the string 'batch' is prefixed to batch values, else Limma complains
    # check if any values are missing the prefix 'batch ' . If yes then add prefix
    batch_var = pd.Series(batch).astype(str)
    batch = batch_var.where(batch_var.str.startswith("batch "), "batch " + batch_var)

    # Convert and assign batch to R as a factor
    globalenv['batch'] = FactorVector(batch)

    # Run removeBatchEffect
    print("Performing Limma batch correction")
    ro.r('expr_corrected <- removeBatchEffect(expr, batch=batch)')
    #ro.r('expr_corrected <- removeBatchEffect(expr, batch=batch, group=condition)')

    # Get the result from R back to Python
    with localconverter(default_converter + pandas2ri.converter):
        expr_limma_corrected = ro.conversion.rpy2py(ro.r['expr_corrected'])

    # Assign row and column names
    expr_limma_corrected = pd.DataFrame(expr_limma_corrected,
                                        columns=ibaq_matrix.columns,
                                        index=ibaq_matrix.index)

    return expr_limma_corrected


# Read ppb ibaq matrix
ibaq_matrix = OTAR_wide.copy()

# Filter missing values in tissue groups
groups_Tissues = ibaq_matrix.columns[ibaq_matrix.columns.str.contains("PXD", regex=True)].str.replace(r'.*_', '', regex=True)

meta_cols = ['targetId', 'datasourceId', 'datatypeId', 'unit', 'targetFromSourceId']
limma_filtered = []

for g in pd.unique(groups_Tissues):
    cols = ibaq_matrix.columns[ibaq_matrix.columns.str.endswith(f"_{g}")]

    # keep only rows (proteins) with less than 20% missing values in each tissue group
    group_mask = ibaq_matrix[cols].isna().mean(axis=1) < 0.20
    ibaq_matrix_filtered = ibaq_matrix.loc[group_mask, cols.union(meta_cols)]

    sample_names = ibaq_matrix_filtered.columns[ibaq_matrix_filtered.columns.str.contains("PXD", regex=True)].tolist()

    # IMPORTANT: Log transform iBAQ values before Batch Effect correction
    ibaq_matrix_filtered = ibaq_matrix_filtered.replace('nan', np.nan)
    ibaq_matrix_filtered[sample_names] = np.log2(ibaq_matrix_filtered[sample_names] + 1)

    ibaq_matrix_filtered = ibaq_matrix_filtered[sample_names].set_index(
        ibaq_matrix_filtered[meta_cols].agg('+'.join,axis=1))

    datasets = pd.Series(sample_names).str.replace(r'-.*', '', regex=True).tolist()
    factorcodes = pd.factorize(datasets)[0] + 1

    batch = ['batch' + str(x) for x in factorcodes]
    tissue = pd.Series(sample_names).str.replace(r'.*_', '', regex=True).tolist()

    # If expression matrix has more than two samples and more than 1 batch
    if (len(ibaq_matrix_filtered.columns) > 2) and (len(set(batch)) > 1):
        # Do limma batch effect correction on each tissue subgroup separately
        limma_corrected_subgroup = limma_batchEffect(ibaq_matrix_filtered, batch)
        limma_filtered.append(limma_corrected_subgroup)
    else:
        limma_filtered.append(ibaq_matrix_filtered)

# Combine all tissue subgroups into one matrix
expr_limma_corrected = pd.concat(limma_filtered, axis=1)

meta = (
    expr_limma_corrected.index.to_series()
    .str.split(r'\+', expand=True)
    .set_axis(meta_cols, axis=1)
)

sample_names = expr_limma_corrected.columns[expr_limma_corrected.columns.str.contains("PXD", regex=True)].tolist()
datasets = pd.Series(sample_names).str.replace(r'-.*', '', regex=True).tolist()
factorcodes = pd.factorize(datasets)[0] + 1
batch = ['batch' + str(x) for x in factorcodes]
tissue = pd.Series(sample_names).str.replace(r'.*_', '', regex=True).tolist()

annotation = pd.DataFrame()
annotation['assayId'] = sample_names
annotation['datasets'] = datasets
annotation['batch'] = batch
annotation['tissue'] = tissue

##########################################
# UMAP clustering
##########################################
def perform_UMAP(inp_expr_df):

    expr_matrix = inp_expr_df.copy()
    # transpose to have rows as samples and columns as features (genes, peptides, etc.).
    expr_matrix_trans = expr_matrix.T
    # change NaN to 0. UMAP does not handle NaN
    expr_matrix_trans[np.isnan(expr_matrix_trans)] = 0
    # Initialize UMAP, Fit and transform
    umap_plotdata = umap.UMAP(n_components=2, random_state=42).fit_transform(expr_matrix_trans)

    umap_plotdata = pd.DataFrame(umap_plotdata,
                                 columns=["UMAP1", "UMAP2"],
                                 index=expr_matrix.columns.tolist())

    umap_plotdata.index.name = 'assayId'
    umap_plotdata = umap_plotdata.reset_index()

    return(umap_plotdata)

umap_plotdata = perform_UMAP(expr_limma_corrected)
umap_plotdata = pd.merge(umap_plotdata, annotation, on='assayId')
umap_plotdata = umap_plotdata.sort_values(by="tissue")

## Plot UMAP
fig1, ax = plt.subplots(figsize=(10,8.5))

sb.scatterplot(
    data=umap_plotdata,
    x="UMAP1",
    y="UMAP2",
    hue="tissue",
    style="datasets",
    ax=ax
)

ax.legend(
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    ncol=2,
    frameon=False
)

plt.tight_layout()

plt.savefig(path+"OTAR_baselinedata_all_umap_plot.pdf")
#plt.show()

####################
# Save UMAP as interactive HTML
fig2 = px.scatter(
    umap_plotdata,
    x="UMAP1",
    y="UMAP2",
    color="tissue",
    hover_data=["datasets", "tissue", "assayId"]
)
fig2.show()
fig2.write_html(path+"OTAR_baselinedata_all_umap_plot.html")

#######################################
# Calculate Zscore for each protein within a sample:
#  i.Calculate Mean of each protein across all samples (population mean)
#  ii.Calculate Standard Deviation of each protein across all samples (population standard deviation)
#  iii. ZScore of protein A in PXD1-1_Heart = (ExpA.PXD1-1_Heart - Population Mean of A)/Population Standard Dev of A. Take absolute values
#######################################
Z_scores = (
    expr_limma_corrected[sample_names]
    .sub(expr_limma_corrected[sample_names].mean(axis=1),axis=0).abs()
    .div(expr_limma_corrected[sample_names].std(axis=1),axis=0)
)

sample_median = expr_limma_corrected.median(axis=0, numeric_only=True)

# Calculate Tissue Specificity Score for each protein within a sample:
#  by dividing Zcore of each protein by median of sample
Tissue_Specificity_scores = Z_scores[sample_names].div(sample_median, axis=1)

#######################################
# Convert matrices from wide to long format
# Tissue Specificity Score matrix
#######################################
Tissue_Specificity_scores = pd.concat(
    [meta, Tissue_Specificity_scores],
    axis=1
)
Tissue_Specificity_scores = Tissue_Specificity_scores.reset_index(drop=True)

# Convert to long format and Merge with initial file + batch corrected expression and TSS
Tissue_Specificity_scores_long = Tissue_Specificity_scores.melt(
    id_vars=meta_cols,
    var_name='donorId',
    value_name='tissueSpecificityScore',
)

Tissue_Specificity_scores_long['tissueBiosampleFromSource'] = Tissue_Specificity_scores_long['donorId'].str.replace(r'.*_', '', regex=True)
Tissue_Specificity_scores_long['donorId'] = Tissue_Specificity_scores_long['donorId'].str.replace(r'_.*', '', regex=True)


#######################################
# Convert matrices from wide to long format
# Batch corrected Expression matrix
#######################################
expr_limma_corrected  = pd.concat(
    [meta, expr_limma_corrected ],
    axis=1
)
expr_limma_corrected  = expr_limma_corrected.reset_index(drop=True)

expr_limma_corrected_long = expr_limma_corrected.melt(
    id_vars=meta_cols,
    var_name='donorId',
    value_name='batchCorrectedExpression')

expr_limma_corrected_long['tissueBiosampleFromSource'] = expr_limma_corrected_long['donorId'].str.replace(r'.*_', '', regex=True)
expr_limma_corrected_long['donorId'] = expr_limma_corrected_long['donorId'].str.replace(r'_.*', '', regex=True)

# Reconvert log2 values back to iBAQ
expr_limma_corrected_long['batchCorrectedExpression'] = (2 ** expr_limma_corrected_long['batchCorrectedExpression']) - 1

# Merge TS and Limma corrected expression dataframes
Limma_TS_merged = pd.merge(expr_limma_corrected_long, Tissue_Specificity_scores_long,
                           on=meta_cols+['donorId', 'tissueBiosampleFromSource'],
                           how='outer')

# Merge Limma_TS dataframe with original OTAR dataframe
All_data_Merged = pd.merge(Limma_TS_merged, OTAR_alldata_clean,
                           on=meta_cols+['donorId', 'tissueBiosampleFromSource'],
                           how='right')
All_data_Merged = All_data_Merged.drop(columns=['donorTissueId'])

# Save All merged data to tab txt file
with open(os.path.join(path,"OTAR_baselinedata_AllData_Long.txt"), "w") as outfile:
    All_data_Merged.to_csv(outfile, sep='\t', index=False)

# Save All merged data to parquet file
All_data_Merged.to_parquet(
    os.path.join(path, "OTAR_baselinedata_AllData.parquet")
)

# Save TSS to tab text file
with open(os.path.join(path, "OTAR_baselinedata_TissueSpecificityScores.txt"), "w") as outfile:
    Tissue_Specificity_scores.to_csv(outfile, sep='\t', index=False)

# Save TSS to parquet file
Tissue_Specificity_scores.to_parquet(
    os.path.join(path, "OTAR_baselinedata_TissueSpecificityScores.parquet")
)

print("End")