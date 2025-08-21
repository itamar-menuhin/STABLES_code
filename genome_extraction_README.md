# Genome Extraction Process

This document describes the process used to extract and construct the genome data file (e.g., `genome.xlsx` or `escCol_genome.csv`) for downstream feature calculation and modeling.

## Data Sources
- **ORF sequences**: Downloaded as FASTA files (e.g., `orf_genomic_all.fasta.gz`) from a relevant genome database (such as NCBI, Ensembl, or SGD).
- **5' UTR sequences**: Downloaded as FASTA or ZIP files (e.g., `SGD_all_ORFs_5prime_UTRs.fsa.zip`).
- **3' UTR sequences**: Downloaded as FASTA or ZIP files (e.g., `SGD_all_ORFs_3prime_UTRs.fsa.zip`).

## Extraction Steps
1. **Parse ORF FASTA**
   - Use Biopython's `SeqIO` to read and index all ORF sequences by gene ID.
2. **Parse UTR FASTA/ZIP**
   - Use Biopython's `SeqIO` to read and index all 5' and 3' UTR sequences by gene ID.
   - For each gene, extract:
     - `gene_id1`, `gene_id2`, `gene_id3` (all set to the gene ID)
     - `gene_prom` (5' UTR sequence)
     - `gene_ORF` (ORF sequence)
     - `gene_UTR3` (3' UTR sequence)
   - Only include genes for which both UTRs and ORF are available.
5. **Save to File**
   - Save the resulting table as an Excel file (`genome.xlsx`) or CSV (`escCol_genome.csv`) using pandas.

## Example Python Code
```python
import pandas as pd
import gzip, zipfile, io
from Bio import SeqIO

# Paths to input files
orf_path = 'orf_genomic_all.fasta.gz'
utr5_zip = 'SGD_all_ORFs_5prime_UTRs.fsa.zip'
utr3_zip = 'SGD_all_ORFs_3prime_UTRs.fsa.zip'
gene_orfs = SeqIO.to_dict(SeqIO.parse(gzip.open(orf_path, 'rt'), 'fasta'))
# Parse UTRs
        gene_utr5 = SeqIO.to_dict(SeqIO.parse(io.TextIOWrapper(f), 'fasta'))
gene_utr3 = {}
    with z.open(z.namelist()[0]) as f:
        gene_utr3 = SeqIO.to_dict(SeqIO.parse(io.TextIOWrapper(f), 'fasta'))

# Build rows
rows = []
for gene_id in gene_orfs:
    prom = next((str(gene_utr5[k].seq) for k in gene_utr5 if gene_id in k), '')
    utr3 = next((str(gene_utr3[k].seq) for k in gene_utr3 if gene_id in k), '')
    if prom and utr3:
        rows.append([gene_id, gene_id, gene_id, prom, str(gene_orfs[gene_id].seq), utr3])

# Save
df = pd.DataFrame(rows, columns=["gene_id1", "gene_id2", "gene_id3", "gene_prom", "gene_ORF", "gene_UTR3"])
df.to_excel('genome.xlsx', index=False)
```

## Notes
- The exact file names and download sources may vary by organism.
- This process ensures all required sequence fields are present for each gene.
- The resulting file is used as input for feature extraction scripts (e.g., `calc_features.py`).
