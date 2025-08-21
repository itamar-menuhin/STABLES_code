

import os

# Clean, minimal version for only the three required files
import os
import gzip
import zipfile
import pandas as pd
from Bio import SeqIO
import io

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))

# File paths
fasta_path = os.path.join(data_dir, "orf_genomic_all.fasta.gz")
utr3_zip = os.path.join(data_dir, "SGD_all_ORFs_3prime_UTRs.fsa.zip")
utr5_zip = os.path.join(data_dir, "SGD_all_ORFs_5prime_UTRs.fsa.zip")

# Parse ORF FASTA
with gzip.open(fasta_path, 'rt') as handle:
    orf_seqs = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))

# Parse 3' UTR FASTA
with zipfile.ZipFile(utr3_zip, 'r') as zipf:
    f3name = zipf.namelist()[0]
    with zipf.open(f3name) as f3:
        with io.TextIOWrapper(f3, encoding='utf-8') as f3_text:
            utr3_seqs = SeqIO.to_dict(SeqIO.parse(f3_text, "fasta"))

# Parse 5' UTR FASTA
with zipfile.ZipFile(utr5_zip, 'r') as zipf:
    f5name = zipf.namelist()[0]
    with zipf.open(f5name) as f5:
        with io.TextIOWrapper(f5, encoding='utf-8') as f5_text:
            utr5_seqs = SeqIO.to_dict(SeqIO.parse(f5_text, "fasta"))

# Build output rows
rows = []
for gene_id in orf_seqs:
    gene_id1 = gene_id
    gene_id2 = gene_id
    gene_id3 = gene_id
    # Find matching 5' UTR key
    gene_prom = ''
    for k in utr5_seqs:
        if gene_id in k:
            gene_prom = str(utr5_seqs[k].seq)
            break
    # Find matching 3' UTR key
    gene_UTR3 = ''
    for k in utr3_seqs:
        if gene_id in k:
            gene_UTR3 = str(utr3_seqs[k].seq)
            break
    gene_ORF = str(orf_seqs[gene_id].seq)
    if gene_prom and gene_UTR3:
        rows.append([gene_id1, gene_id2, gene_id3, gene_prom, gene_ORF, gene_UTR3])

output_path = os.path.join(data_dir, "genome2.xlsx")
df = pd.DataFrame(rows, columns=["gene_id1", "gene_id2", "gene_id3", "gene_prom", "gene_ORF", "gene_UTR3"])
df.to_excel(output_path, index=False)
print(f"genome2.xlsx created successfully at {output_path}")

