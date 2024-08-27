# Description of OT formats


## Aggregated view 
Aggregated protein abundance of each protein across samples within a sample group. Fields comprise

**"geneProductId"**: <string> Ensembl gene accessions ex. "ENSG00000000003",

**"expression"**: <list>      containing dictionaries quantile normalised protein abundaces in each group

              [      
                {
                  "assayGroupId": <string> - name of assay group to which some samples belong to,
                  "min": <float> - minimum protein abundance across all samples,
                  "q1": <float>  - first quartile abundance,
                  "q2": <float>  - second quartile abundance,
                  "q3": <float>  - third quartile abundance,
                  "max": <float> - maximum protein abundance across all samples
                }
              ]
              
## Sample view
Protein abundances expressed for each protein in each sample. Fields comprise-

  **"geneProductId"**: <string> - Ensembl gene accessions ex. "ENSG00000000003",
  
  **"unit"**: <string>          - unit of protein expression ex. "ppb",
  
  **"expression"**: <list>      - containing dictionaries of sample names & protein abundances
  
              [
                {
                  "assayId": <string> - name of sample (no defined format),
                   "value": <float>   - expression value
                }
              ]
              
## Metadata
Meatadata of samples. Fields comprise-

  **"experimentId"**: <string>   - PRIDE dataset accession  ex. "PXD012345",
  
  **"experimentType"**: <string> - type of proteomics study ex. "proteomics_baseline" or "proteomics_differential",
  
  **"species"**: <string>        - CV term taxonomic name   ex. "Mus musculus" or "Homo sapiens",
  
  **"speciesOntURI"**: <string>  - NCBI URL of species      ex. "http://purl.obolibrary.org/obo/NCBITaxon_10090",
  
  **"pubmedIds"**: <list>        - list of PubMed ids <integer> listing this study
  
                              [
                                "22955982",
                                "22955987"
                              ],
  **"provider"**: <string>           - semicolon (;) delimited names of anuthors ex. "Jose; Gonzalez; Anne-Maud",
  
  **"experimentalDesigns"**: <list>  - list of dictionaries containing metadata of each assay group
  
                [
                  {
                      "assayGroupId": <string>       - ex. "g1",
                      "assayId": <string>            - ex. "ERR1361600",
                      "assayGroup": <string>         - ex. "brain",
                      "age": <string>                - ex. "8 week",
                      "organismPart": <string>       - CV term  ex. "brain",
                      "organismPartOntURI": <string> - ex. "http://purl.obolibrary.org/obo/UBERON_0000955",
                      "sex": <string>                - CV term  ex. "male",
                      "sexOntURI": <string>          - ex. "http://purl.obolibrary.org/obo/PATO_0000384",
                      "strain": <string>             - ex. "C57BL/6",
                      "strainOntURI": <string>       - ex. "http://www.ebi.ac.uk/efo/EFO_0004472"
                  },
                ]
