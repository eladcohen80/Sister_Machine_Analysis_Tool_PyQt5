import os
import pandas as pd

PARAMETERS = ["X0", "Xf", "Alpha", "Tau_Min", "F_ratio", "Phi", "Delta", "R"]


def _series_to_row(section_name, file_name, values):
    row = {"Section": section_name, "File": file_name}
    for col in PARAMETERS:
        row[col] = values.get(col, float("nan"))
    return row


def process_cycles_summary(folder_path):
    cycle_files = sorted(
        [
            filename
            for filename in os.listdir(folder_path)
            if filename.endswith("_cycles.csv.csv")
        ]
    )

    if not cycle_files:
        raise ValueError("No _cycles.csv.csv files were found in the selected folder")

    all_cycle_values = []
    per_file_global = []
    per_file_side_minus_global = []

    for filename in cycle_files:
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        missing_columns = [col for col in (["Side"] + PARAMETERS) if col not in df.columns]
        if missing_columns:
            print(f"Skipping {filename}: missing columns {missing_columns}")
            continue

        side_norm = df["Side"].astype(str).str.strip().str.lower()

        cycle_df = df[side_norm.isin(["left", "right"])].copy()
        if not cycle_df.empty:
            for col in PARAMETERS:
                cycle_df[col] = pd.to_numeric(cycle_df[col], errors="coerce")
            all_cycle_values.append(cycle_df[PARAMETERS])

        summary_df = df[side_norm == "summary"].copy()
        if summary_df.empty:
            print(f"Skipping {filename}: no SUMMARY rows from step3")
            continue

        summary_cycle_norm = summary_df["Cycle"].astype(str).str.strip()

        for col in PARAMETERS:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

        grand_mean_row = summary_df[summary_cycle_norm == "Grand_Mean"]
        left_minus_row = summary_df[summary_cycle_norm == "Left_Minus_Grand"]
        right_minus_row = summary_df[summary_cycle_norm == "Right_Minus_Grand"]

        if grand_mean_row.empty or left_minus_row.empty or right_minus_row.empty:
            print(f"Skipping {filename}: missing one or more required SUMMARY rows")
            continue

        per_file_global.append(grand_mean_row.iloc[0][PARAMETERS])
        per_file_side_minus_global.append(left_minus_row.iloc[0][PARAMETERS])
        per_file_side_minus_global.append(right_minus_row.iloc[0][PARAMETERS])

    if not per_file_global:
        raise ValueError("No valid SUMMARY rows were found in _cycles.csv.csv files")

    if all_cycle_values:
        all_values_df = pd.concat(all_cycle_values, ignore_index=True)
        variance_all_values = all_values_df[PARAMETERS].var(ddof=1)
    else:
        variance_all_values = pd.Series({col: float("nan") for col in PARAMETERS})

    variance_file_mean_global = pd.DataFrame(per_file_global)[PARAMETERS].var(ddof=1)
    variance_file_side_minus_global = pd.DataFrame(per_file_side_minus_global)[PARAMETERS].var(ddof=1)

    summary_rows = [
        _series_to_row("1_Variance_All_Values", "ALL_FILES", variance_all_values),
        _series_to_row("7_Variance_Of_File_Global_Mean", "ALL_FILES", variance_file_mean_global),
        _series_to_row("8_Variance_Of_Side_Minus_Global", "ALL_FILES", variance_file_side_minus_global),
    ]

    final_df = pd.DataFrame(summary_rows)
    output_file = os.path.join(folder_path, "cycles_summary_statistics.csv")
    final_df.to_csv(output_file, index=False)

    print(f"Saved summary statistics: {output_file}")
    return output_file


if __name__ == "__main__":
    from tkinter import filedialog

    analysis_dir = filedialog.askdirectory(title="Select folder containing _cycles.csv.csv files")
    if analysis_dir:
        process_cycles_summary(analysis_dir)
