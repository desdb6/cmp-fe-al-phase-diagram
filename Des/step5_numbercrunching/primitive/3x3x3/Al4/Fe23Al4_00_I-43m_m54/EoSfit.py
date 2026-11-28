import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from calculations import *
import pandas as pd

df = pd.read_csv(r"Des\step5_numbercrunching\primitive\3x3x3\Al4\Fe23Al4_00_I-43m_m54\ev_curve.csv")
eos_fit = EoSfit(df["energy"], df["volume"], output_path=r"Des\step5_numbercrunching\primitive\3x3x3\Al4\Fe23Al4_00_I-43m_m54\fit_results")