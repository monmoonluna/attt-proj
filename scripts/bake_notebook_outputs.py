#!/usr/bin/env python3
"""
scripts/bake_notebook_outputs.py
Chạy và lưu kết quả (pre-computed outputs) vào reports/demo_ket_qua.ipynb
mà không phụ thuộc vào Jupyter socket / ZMQ (hoạt động 100% trong sandbox).
"""

import os
import sys
import io
import json
import base64
import contextlib
import nbformat as nbf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(ROOT_DIR, "reports", "demo_ket_qua.ipynb")

def bake():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    # Context to execute cells
    exec_globals = {
        "__file__": NOTEBOOK_PATH,
        "ROOT_DIR": ROOT_DIR,
        "plt": plt,
        "pd": pd,
        "np": np,
    }

    # Switch working directory to reports/ during execution
    old_cwd = os.getcwd()
    reports_dir = os.path.join(ROOT_DIR, "reports")
    os.chdir(reports_dir)

    execution_count = 1

    try:
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            source = cell.source
            stdout_capture = io.StringIO()
            captured_outputs = []

            # Custom display function
            def custom_display(obj):
                if isinstance(obj, pd.DataFrame):
                    captured_outputs.append(nbf.v4.new_output(
                        output_type="display_data",
                        data={
                            "text/html": obj.to_html(classes="dataframe", border=0),
                            "text/plain": obj.to_string()
                        }
                    ))
                elif hasattr(obj, "data") and isinstance(obj.data, (bytes, str)):
                    # Image object
                    ext = getattr(obj, "format", "png") or "png"
                    data_b64 = base64.b64encode(obj.data if isinstance(obj.data, bytes) else obj.data.encode()).decode("utf-8")
                    captured_outputs.append(nbf.v4.new_output(
                        output_type="display_data",
                        data={
                            f"image/{ext}": data_b64,
                            "text/plain": f"<IPython.core.display.Image object>"
                        }
                    ))
                elif hasattr(obj, "filename") and obj.filename and os.path.exists(obj.filename):
                    with open(obj.filename, "rb") as img_f:
                        data_b64 = base64.b64encode(img_f.read()).decode("utf-8")
                    ext = os.path.splitext(obj.filename)[1].replace(".", "") or "png"
                    captured_outputs.append(nbf.v4.new_output(
                        output_type="display_data",
                        data={
                            f"image/{ext}": data_b64,
                            "text/plain": f"<IPython.core.display.Image filename='{obj.filename}'>"
                        }
                    ))

            exec_globals["display"] = custom_display

            # Intercept plt.show()
            def custom_plt_show():
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("utf-8")
                plt.close()
                captured_outputs.append(nbf.v4.new_output(
                    output_type="display_data",
                    data={
                        "image/png": b64,
                        "text/plain": "<Figure size ... with ... Axes>"
                    }
                ))

            exec_globals["plt"].show = custom_plt_show

            # Execute cell
            try:
                with contextlib.redirect_stdout(stdout_capture):
                    exec(source, exec_globals)
            except Exception as e:
                captured_outputs.append(nbf.v4.new_output(
                    output_type="error",
                    ename=type(e).__name__,
                    evalue=str(e),
                    traceback=[f"Cell execution error: {e}"]
                ))

            out_text = stdout_capture.getvalue()
            if out_text:
                captured_outputs.insert(0, nbf.v4.new_output(
                    output_type="stream",
                    name="stdout",
                    text=out_text
                ))

            cell.outputs = captured_outputs
            cell.execution_count = execution_count
            execution_count += 1

    finally:
        os.chdir(old_cwd)

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"✓ Hoàn tất nướng kết quả vào {NOTEBOOK_PATH} thành công!")

if __name__ == "__main__":
    bake()
