# tests/test_flow.py
import numpy as np
from openpyxl import Workbook
from cc_reservoir.io.flow import read_time_cc_td

def test_reads_time_cc_td_block(tmp_path):
    wb = Workbook(); ws = wb.active
    ws.append(["Time", "CC", "TD"])
    for row in [[0, 84.7, 76.4], [59, 116.8, 75.6], [119, 270.4, 100.6]]:
        ws.append(row)
    p = tmp_path / "s.xlsx"; wb.save(p)
    t, cc, td = read_time_cc_td(str(p), sheet=ws.title, header_row=1)
    assert np.allclose(t, [0, 59, 119])
    assert np.allclose(cc, [84.7, 116.8, 270.4])
    assert np.allclose(td, [76.4, 75.6, 100.6])
