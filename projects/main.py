# %%
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import GEOparse

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Align import PairwiseAligner, substitution_matrices

from biotite.sequence import NucleotideSequence, ProteinSequence, CodonTable
import biotite.sequence.io.fasta as fasta

import seaborn as sns
import matplotlib.pyplot as plt
# %%
# Task 1 - Data Retrieval
gse = GEOparse.get_GEO(geo="GSE10245", destdir="./")

# %%
# Non-small cell lung cancer (NSCLC) can be classified into the major subtypes 
# adenocarcinoma (AC) and squamous cell carcinoma (SCC) subtypes. 
# This file has global gene expression profiling of 58 human high grade NSCLC specimens.

#%%
data = gse.pivot_samples('VALUE')

adeno_samples = []
squamous_samples = []

for gsm_id, gsm in gse.gsms.items():
    status = gsm.metadata['characteristics_ch1'][0]
    if 'adenocarcinoma' in status:
        adeno_samples.append(gsm_id)
    elif 'squamous cell carcinoma' in status:
        squamous_samples.append(gsm_id)

# %%
adeno_mean = data[adeno_samples].mean(axis=1)
squamous_mean = data[squamous_samples].mean(axis=1)
adeno_sd = data[adeno_samples].std(axis=1)
squamous_sd = data[squamous_samples].std(axis=1)

# %%
results = pd.DataFrame({
    'adeno_mean': adeno_mean,
    'squamous_mean': squamous_mean,
    'adeno_sd': adeno_sd,
    'squamous_sd': squamous_sd
})
results

results['score'] = (adeno_mean - squamous_mean) / (adeno_sd + squamous_sd)
top_up = results.sort_values('score', ascending=False).head(1)
top_down = results.sort_values('score', ascending=True).head(1)

# %%
gene_1_id = top_down.reset_index().ID_REF[0]
gene_2_id = top_up.reset_index().ID_REF[0]

# %%
genes_to_plot = [gene_1_id, gene_2_id]
plot_df = data.loc[genes_to_plot].T.reset_index()

# %%
plot_df['status'] = np.select([plot_df['name'].isin(adeno_samples),plot_df['name'].isin(squamous_samples)], ['A','S'], default='None')

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=plot_df, x='status', y=gene_1_id)
plt.title("Gene Expression Comparison: Adeno vs Squamous")
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=plot_df, x='status', y=gene_2_id)
plt.title("Gene Expression Comparison: Adeno vs Squamous")
plt.show()

# %%
gpl = gse.gpls["GPL570"].table
gene_1_symbol = gpl[gpl['ID'] == gene_1_id].reset_index(drop = True)["Gene Symbol"][0]
gene_2_symbol = gpl[gpl['ID'] == gene_2_id].reset_index(drop = True)["Gene Symbol"][0]

# %%
# Gene 1 Symbol: DSC3
# Gene 2 Symbol: CGN

#%%
# Task 2 - Sequence Extraction
# Downladed coding seqences as fasta files form:
# DSC3: https://www.ncbi.nlm.nih.gov/datasets/gene/1825/
# CGN: https://www.ncbi.nlm.nih.gov/datasets/gene/57530/

# %%
gene_1_file = fasta.FastaFile.read("./DSC3.fasta")
gene_2_file = fasta.FastaFile.read("./CGN.fasta")

gene_1_seq = list(fasta.get_sequences(gene_1_file).values())[0]
gene_2_seq = list(fasta.get_sequences(gene_2_file).values())[0]

# %%
# Task 3 - Translation to Protein
codon_table = {
    'A': ['GCT','GCC','GCA','GCG'],
    'C': ['TGT','TGC'],
    'D': ['GAT','GAC'],
    'E': ['GAA','GAG'],
    'F': ['TTT','TTC'],
    'G': ['GGT','GGC','GGA','GGG'],
    'H': ['CAT','CAC'],
    'I': ['ATT','ATC','ATA'],
    'K': ['AAA','AAG'],
    'L': ['TTA','TTG','CTT','CTC','CTA','CTG'],
    'M': ['ATG'],
    'N': ['AAT','AAC'],
    'P': ['CCT','CCC','CCA','CCG'],
    'Q': ['CAA','CAG'],
    'R': ['CGT','CGC','CGA','CGG','AGA','AGG'],
    'S': ['TCT','TCC','TCA','TCG','AGT','AGC'],
    'T': ['ACT','ACC','ACA','ACG'],
    'V': ['GTT','GTC','GTA','GTG'],
    'W': ['TGG'],
    'Y': ['TAT','TAC'],
    '*': ['TAA','TAG','TGA']
}

reverse_codon_table = {i: k for k, v in codon_table.items() for i in v}
new_codon_table = CodonTable(
    codon_dict = reverse_codon_table,
    starts=['ATG']
)

gene_1_prot_list = gene_1_seq.translate(codon_table=new_codon_table, met_start=False)
gene_2_prot_list = gene_2_seq.translate(codon_table=new_codon_table, met_start=False)

# Selecting the longest ORF for each gene
best_prot_1 = max(gene_1_prot_list, key=len)
best_prot_2 = max(gene_2_prot_list, key=len)

protein_fasta = fasta.FastaFile()

protein_fasta["DSC3_protein_translated"] = str(best_prot_1)
protein_fasta["CGN_protein_translated"] = str(best_prot_2)

protein_fasta.write("translated_proteins.fasta")

# %%
with open("DSC3_prot.fasta", "w") as f:
    f.write(gene_1_prot)

with open("CGN_prot.fasta", "w") as f:
    f.write(gene_2_prot)
# %%
prot = "MHTRLKYSILQQTPRSPGLFSVHPSTGVITTVSHYLDREVVDKYSLIMKVQDMDGQFFGLIGTSTCIITVTDSNDNAPTFRQNAYEAFVEENAFNVEILRIPIEDKDLINTANWRVNFTILKGNENGHFKISTDKETNEGVLSVVKPLNYEENRQVNLEIGVNNEAPFARDIPRVTALNRALVTVHVRDLDEGPECTPAAQYVRIKENLAVGSKINGYKAYDPENRNGNGLRYKKLHDPKGWITIDEISGSIITSKILDREVETPKNELYNITVLAIDKDDRSCTGTLAVNIEDVNDNPPEILQEYVVICKPKMGYTDILAVDPDEPVHGAPFYFSLPNTSPEISRLWSLTKVNDTAARLSYQKNAGFQEYTIPITVKDRAGQAATKLLRVNLCECTHPTQCRATSRSTGVILGKWAILAILLGIALLFSVLLTLVCGVFGATKGKRFPEDLAQQNLIISNTEAPGDDRVCSANGFMTQTTNNSSQGFCGTMGSGMKNGGQETIEMMKGGNQTLESCRGAGHHHTLDSCRGGHTEVDNCRYTYSEWHSFTQPRLGEKLHRCNQNEDRMPSQDYVLTYNYEGRGSPAGSVGCCSEKQEEDGLDFLNNLEPKFITLAEACTKR"

# %%
result_handle = NCBIWWW.qblast('blastp', 'swissprot', prot)
blast_record = NCBIXML.read(result_handle)

# %%
df = pd.DataFrame(
    ([
        hsp.identities / hsp.align_length * 100 if hsp.align_length > 0 else 0,
        hsp.expect,
        hsp.score,
        hsp.align_length / len(prot) * 100,
        alignment.hit_def,
        hsp.sbjct,
    ] for alignment in blast_record.alignments for hsp in alignment.hsps),
    columns=["identity", "e-value", "bit score", "coverage", "description", "sbjct"]
)

# %%
# Task 4 - top 10 BLASTp hits
df.iloc[:10, :].loc[:, ["identity", "e-value", "description"]]

# %%
print(*df["description"][:10], sep="\n")

# %%
matrix  = substitution_matrices.load("BLOSUM62")

aligner = PairwiseAligner()
aligner.mode = "global"

aligner.substitution_matrix = matrix
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5

# %%
# Task 5 - top 5 alignments
for sbjct in df.loc[:, "sbjct"].iloc[:5]:
    alignments = aligner.align(prot, sbjct)
    print(alignments[0])

# %%
n = min(df.shape[0], 100) + 1
score_matrix = np.zeros((n, n))
proteins = [prot, *df["sbjct"][:n].str.replace("-", "")]

# %%
for i, a in enumerate(proteins):
    for j, b in enumerate(proteins[i:], start=i):
        score = aligner.score(a, b)
        score_matrix[i, j] = score
        score_matrix[j, i] = score

# %%
score_matrix /= np.linalg.norm(score_matrix, axis=0)

# %%
reduced = PCA(n_components=2).fit_transform(score_matrix)

# %%
k_list = np.arange(2, 10)
cluster_scores = np.zeros((10, k_list.shape[0]))
for i in range(10):
    for j, k in enumerate(k_list):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(reduced)
        score = silhouette_score(reduced, labels)
        cluster_scores[i, j] = score

plt.plot(np.mean(cluster_scores, axis=0))
plt.xticks(np.arange(k_list.shape[0]), k_list);

# %%
k = 3
kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(reduced)
score = silhouette_score(reduced, labels)
k, score

# %%
# Task 6 - plot after dimensionality reduction & clustering
plt.scatter(reduced[1:,0], reduced[1:,1], c=labels[1:], marker="x")
plt.scatter(reduced[:1,0], reduced[:1,1], c="None", edgecolors="red", marker="o", s=200)

plt.show()
