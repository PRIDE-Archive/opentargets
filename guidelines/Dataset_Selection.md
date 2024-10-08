### Criteria for selecting datasets to be reanalysed

PRIDE has many datasets for each biological condition of interest, and therefore a selection strategy must be employed to pick the most ‘interesting’ or ‘useful’ good quality datasets that are suitable for reanalysis.

To select proteomics datasets for reanalysis, we propose the following criteria:

- First prioritise datasets related to diseases suggested by industry partners as well as represented by RNAseq experiments in Open Targets.
- Query PRIDE and ProteomeCentral databases with disease names and 'Homo sapiens' as keywords. 
- Each dataset will be checked, in their respective publications, for below attributes that are considered important or desirable for data reanalysis. 
- Each attribute is given a weight to denote their importance. The weights are from 1 to 3, negative weights are added to denote undesirability.
  - **1**: not very important, but useful; 
  - **2**: important, improves data quality;
  - **3**: very important, will affect data quality.

The attributes included in evaluating a dataset for selectability for reanalysis are:

- **(a)** Number of raw files [**3**]: Importance is given to datasets that have a large number of samples/MS runs. An arbitrary number of 50 raw files are deemed as 'large dataset,’ any dataset with 50 or more raw files is given 3 points.
- **(b)** Submission date [**1**]: Datasets that are more recent are given importance. Datasets submitted within the last 5 years are given 1 point.
- **(c)** Sample metadata [**3**]: Sample metadata links sample characteristics like age, gender, treatment, clinical diagnosis, etc. with raw MS files. This information is present in manuscript tables or supplementary tables or submitted to PRIDE/ProteomeCentral during dataset submission. If metadata is not available, it can be obtained by contacting authors of the respective manuscript. 
- **(d)** Clear diagnosis [**3**]: The samples should have clear clinical diagnosis.
- **(e)** Confounders [**-3**]: Confounders such as other existing medical conditions other than the one investigated, for example, undergoing medical treatment or time course or induced perturbations or microbe interactions, etc. The samples should not have any such confounders that could influence data quality.
- **(f)** Tissue [**1**]: Samples should come from tissues, not from fluid samples or cell lines or cell cultures. Although not critical for data analysis, we have in our previous analysis looked at tissue samples only.
- **(g)** Serum/plasma/fuids: Samples collected from serum or plasma or fluid samples like CSF, saliva, urine, for datasets with:
   - Differential expression [**2**]: We can have differential expression datasets using fluids for different diseases.
   - Baseline expression [**-3**]: We are not accepting fluids for baseline expression.  
- **(h)** Cell lines/cells [**-3**]: Samples should not come from cell lines or cell cultures.
- **(i)** Biological replicates [**3**]: Samples from more than 1 individual are critical for statistical analysis, therefore, biological replicates are important. 
- **(j)** Technical replicates [**3**]: Samples should have technical replicates and is critical for statistical analysis. 
- **(k)** Analytical method: 
   - DDA Label-free [**1**]: Data should come from label-free experiments.
   - DDA Label methods [**2**]: Data should come from TMT/iTRAQ/SILAC experiments. These experiments have better quantification than label-free experiments.
   - DIA Label-free [**3**]: Data should come from DIA label-free experiments. These experiments have better quantification than DDA label-free experiments. 
- **(l)** Fractionated [**2**]: Fractionated datasets have in general more deep coverage than non-fractionated experiments. 
- (**m**) PTM-enriched [**-3**]: Datasets should not be enriched for post-translational modifications. 
- **(n)** Knock-out [**-3**]: Datasets should not come from knock-out experiements (perturbations). 
- (**o**) Enriched or immuno-pull downs [**-3**]: Datasets should not come from experiments enriched for certain proteins from immuno pulldown assays. 
- **(p)** Multiomics [**3**]: It is desirable if datasets have samples analysed for both proteomics and transcriptomics (RNAseq) experiments. 

A threshold of selection score should be determined to consider which datasets to consider for downstream reanalysis. 

### Exisiting dataset curation

https://docs.google.com/spreadsheets/d/1vkXsz2m1qLkzWzt6MgQYZEGgUPg3-QjxgWK6XLKPCFc/edit?gid=0#gid=0
