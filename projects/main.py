# %%
import pandas as pd

from Bio.Blast import NCBIWWW, NCBIXML
from Bio import pairwise2
from Bio.Align import PairwiseAligner

# %%
prot = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV"

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
