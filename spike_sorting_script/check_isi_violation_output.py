import spikeinterface.full as si
import pandas as pd
from pathlib import Path

analyzer_path = Path("/share/home/mitan/Cuhksz4090/spike_sorting/mountainsort4/sorting_results_session2_v2/Amygdala/analyzer")
csv_path = Path("/share/home/mitan/Cuhksz4090/spike_sorting/mountainsort4/sorting_results_session2_v2/all_regions_units_summary.csv")

an = si.load_sorting_analyzer(str(analyzer_path))
qm = an.get_extension("quality_metrics").get_data()

print("QM columns:", list(qm.columns))
for c in qm.columns:
    if "isi" in c.lower():
        print(c, "non-null:", qm[c].notna().sum(), "total:", len(qm))

df = pd.read_csv(csv_path)
print("CSV isi_violation non-null:", df["isi_violation"].notna().sum(), "total:", len(df))