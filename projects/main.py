# %%
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Align import PairwiseAligner, substitution_matrices

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
