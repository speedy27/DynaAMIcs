import sys, collections
import pyarrow.parquet as pq
p = sys.argv[1]
pf = pq.ParquetFile(p)
print("rows", pf.metadata.num_rows, "rowgroups", pf.num_row_groups, "cols", pf.schema_arrow.names)
df = pf.read_row_group(0, columns=["drug", "cell_line"]).to_pandas()
print("cell_lines(rg0):", df["cell_line"].nunique(), " drugs(rg0):", df["drug"].nunique())
c = collections.Counter(df["drug"])
ctrl = [d for d in c if "DMSO" in str(d).upper() or "control" in str(d).lower() or "vehicle" in str(d).lower()]
print("CONTROLS:", [(d, c[d]) for d in ctrl[:5]])
print("top drugs:", c.most_common(5))
