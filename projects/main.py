# %%
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import GEOparse

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Align import PairwiseAligner, substitution_matrices

from biotite.sequence import NucleotideSequence, ProteinSequence, CodonTable
import biotite.sequence.io.fasta as fasta
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics

from umap import UMAP
# %% [markdown]
# # Task 1 - Data Retrieval
# %%
gse = GEOparse.get_GEO(geo="GSE10245", destdir="./")

# %% [markdown]
# Accession: GSE10245
#
# Organism: Homo sapiens (Human)
#
# Condition: Adenocarcinoma vs. Squamous Cell Carcinoma
#
# The dataset compares two major histological subtypes: Adenocarcinoma and Squamous Cell Carcinoma. The data contains 54,675 features normalized via the gcRMA method, which produces values on a log2​ scale.
# %%
data = gse.pivot_samples('VALUE')

print(f"Features: {data.shape[0]}")
print(f"Samples: {data.shape[1]}")
print(f"Global Mean: {data.values.mean()}")
print(f"Global Std: {data.values.std()}")
print(f"Min: {data.values.min()}")
print(f"Max: {data.values.max()}")

# %% [markdown]
# The dataset consists of a matrix of 58 samples. The global expression values have a mean of approx. 4.73 and a range of approx. 0.64 to 15.62. The standard deviation of the dataset is approx. 2.87.
 
# %%
adeno_samples = []
squamous_samples = []

for gsm_id, gsm in gse.gsms.items():
    status = gsm.metadata['characteristics_ch1'][0]
    if 'adenocarcinoma' in status:
        adeno_samples.append(gsm_id)
    elif 'squamous cell carcinoma' in status:
        squamous_samples.append(gsm_id)

# %% [markdown]
# Looking for genes with a high mean difference and high consistency within the two cancer subtypes.

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

# %% [markdown]
#The Adeno group shows minimal expression with very low variance, while the Squamous group shows significantly higher expression. Considering that the gene expression is given on log2​ scale, the mean difference approximately 20 times higher in expression for Squamous cells.

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=plot_df, x='status', y=gene_2_id)
plt.title("Gene Expression Comparison: Adeno vs Squamous")
plt.show()

# %% [markdown]
# In this case expression levels in the Squamos group are significantly lower compared to the Adeno group. Although the Squamous group has higher internal variation, the "boxes" of the two distributions do not overlap indicating a siginicant difference in expression.
# %%
gpl = gse.gpls["GPL570"].table
gene_1_symbol = gpl[gpl['ID'] == gene_1_id].reset_index(drop = True)["Gene Symbol"][0]
gene_2_symbol = gpl[gpl['ID'] == gene_2_id].reset_index(drop = True)["Gene Symbol"][0]

# %% [markdown]
# Gene 1 Symbol: DSC3
#
# Gene 2 Symbol: CGN
#
# DSC3, known as Desmocollin 3 found primarily in epithelial cells where they constitute the adhesive proteins of the desmosome cell-cell junction and are required for cell adhesion and desmosome formation.
#
# CGN, known as Cingulin enables cadherin binding activity. Predicted to be involved in microtubule cytoskeleton organization. Predicted to act upstream of or within bicellular tight junction assembly and epithelial cell morphogenesis. Located in bicellular tight junction and plasma membrane.
# %% [markdown]
# # Task 2 - Sequence Extraction
# Downladed coding seqences as fasta files form:
# 
# DSC3: https://www.ncbi.nlm.nih.gov/datasets/gene/1825/
# 
# CGN: https://www.ncbi.nlm.nih.gov/datasets/gene/57530/

# %%
gene_1_file = fasta.FastaFile.read("./DSC3.fasta")
gene_2_file = fasta.FastaFile.read("./CGN.fasta")

gene_1_seq = list(fasta.get_sequences(gene_1_file).values())[0]
gene_2_seq = list(fasta.get_sequences(gene_2_file).values())[0]

# %% [markdown]
# # Task 3 - Translation to Protein
# %%
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

gene_1_prot_list = gene_1_seq.translate(codon_table=new_codon_table, met_start=False)[0]
gene_2_prot_list = gene_2_seq.translate(codon_table=new_codon_table, met_start=False)[0]

# Selecting the longest ORF for each gene
best_prot_1 = max(gene_1_prot_list, key=len)
best_prot_2 = max(gene_2_prot_list, key=len)

protein_fasta = fasta.FastaFile()

protein_fasta["DSC3_protein_translated"] = str(best_prot_1)
protein_fasta["CGN_protein_translated"] = str(best_prot_2)

protein_fasta.write("translated_proteins.fasta")

# %%
prot_1 = str(best_prot_1)
prot_2 = str(best_prot_2)

# %% [markdown]
# # Task 4 - top 10 BLASTp hits
# %%
def get_blast_hits(prot: str):
    result_handle = NCBIWWW.qblast('blastp', 'swissprot', prot, hitlist_size=100)
    return NCBIXML.read(result_handle)

# %%
blast_1 = get_blast_hits(prot_1)

# %%
blast_2 = get_blast_hits(prot_2)
# apparently there are only 47 matches

# %%
prot = prot_1
blast_record = blast_1

# %%
prot = prot_2
blast_record = blast_2

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
df.iloc[:10, :].loc[:, ["identity", "e-value", "description"]]

# %%
print(*df["description"][:10], sep="\n")

# %% [markdown]
# # Predicted protein functions
# ## Protein 1 (DSC3)
# Predicted desmocollin (cadherin family) protein, most similar to desmocollin-3 and related isoforms (desmocollin-1/2). Likely a calcium-dependent transmembrane cell adhesion glycoprotein that functions as a core component of desmosomes, mediating intercellular adhesion through interactions with other desmosomal cadherins. Involved in maintaining tissue integrity and mechanical stability, particularly in epithelial tissues.
# ## Protein 2 (CGN)
# Predicted cingulin protein, a cytoplasmic component of tight junctions. Likely functions as a scaffolding protein linking transmembrane tight junction proteins to the actin cytoskeleton and regulating junction assembly and barrier function. Involved in maintaining epithelial cell polarity, cell–cell adhesion, and paracellular permeability.

# %%
matrix  = substitution_matrices.load("BLOSUM62")

aligner = PairwiseAligner()
aligner.mode = "global"

aligner.substitution_matrix = matrix
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5

# %% [markdown]
# # Task 5 - top 5 alignments

# %%
for sbjct in df.loc[:, "sbjct"].iloc[:5].str.replace("-", ""):
    alignments = aligner.align(prot, sbjct)
    print(alignments[0])

# %%
others = list(df.loc[:, "sbjct"].iloc[:3])
matrix = np.ones((len(prot), max(map(len, others)), 3))

for channel, other in enumerate(others):

    for i in range(len(prot)):
        for j in range(len(other)):
            if prot[i] == other[j]:
                matrix[i, j, channel] = 0

plt.imshow(matrix, origin="lower")
plt.xlabel("Original protein")
plt.ylabel("Top 3 matches from BLASTp (as 3 RGB channels)")
plt.title("RGB Dot Plot")
plt.show()

# %% [markdown]
# # Alignment – comments
# Both proteins have very good global matches, and even a ~100% match each, so we don't have to do local matching. Protein 2 has a few >90% matches, while protein 1 only has <80% matches after the best one.
#
# We use the default global alignment algorithm, because we have near full matches, and the standard BLOSUM62 matrix.

# %%
n = min(df.shape[0], 100) + 1
print(n)
score_matrix = np.zeros((n, n))
proteins = [prot, *df["sbjct"][:n-1].str.replace("-", "")]

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
reduced = UMAP().fit_transform(score_matrix)

# %%
k_list = np.arange(2, 10)
cluster_scores = np.zeros((10, k_list.shape[0]))
for i in range(10):
    for j, k in enumerate(k_list):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(reduced)
        score = silhouette_score(reduced, labels)
        cluster_scores[i, j] = score

cluster_scores = np.mean(cluster_scores, axis=0)
plt.plot(cluster_scores)
plt.title("Silhouette score vs number of clusters")
plt.xlabel("number of clusters (k)")
plt.ylabel("silhouette score")
plt.xticks(np.arange(k_list.shape[0]), k_list);

# %%
k = int(k_list[np.argmax(cluster_scores)])
# kmeans is more than enough for this task, if we give it the right number of clusters
kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(reduced)
score = silhouette_score(reduced, labels)
print(f"number of clusters: {k}, silhouette score: {score:.2f}")

# %% [markdown]
# # Task 6 - plot after dimensionality reduction & clustering
# %%
plt.scatter(reduced[:,0], reduced[:,1], c=labels, marker="x")
plt.scatter(reduced[:1,0], reduced[:1,1], c="None", edgecolors="red", marker="o", s=200)
plt.title("Protein pairwise similarities after dimensionality reduction & clustering")
plt.xticks([])
plt.yticks([])
plt.text(reduced[0,0] + 1, reduced[0,1], "Original protein", color="red")

# %% [markdown]
# PCA doesn't give satisfying results. UMAP does, and is better than t-SNE.
# K-means is a simple clustering algorithm, but gives good results when given a good number of clusters.
# By testing a range of cluster numbers and averaging over multiple runs,
# we can reliably automatically find the visually optimal number of clusters.

# %%

# %% [markdown]
# ### Authors:
# Paweł Buczyński
# 
# Jagna Wnuk
# %%
