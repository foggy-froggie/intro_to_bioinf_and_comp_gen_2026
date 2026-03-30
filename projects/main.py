# %%
import pandas as pd

from Bio.Blast import NCBIWWW, NCBIXML

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
    ] for alignment in blast_record.alignments for hsp in alignment.hsps),
    columns=["identity", "e-value", "bit score", "coverage", "description"]
)

# %%
# Task 4 - top 10 BLASTp hits
df.iloc[:10, :].loc[:, ["identity", "e-value", "description"]]
