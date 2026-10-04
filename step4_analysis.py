import os
import pandas as pd
import csv
import numpy as np
# try:
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
# except ModuleNotFoundError:
#     plt = None

TIME_STEP_MINUTES = 3

def analyze_cycles_robust(lengths, side_name, filename_base):
    cycles = []
    # מזהה חלוקה לפי צניחה של 30% באורך
    division_frames = [
        i
        for i in range(len(lengths) - 1)
        if lengths[i] > 0 and (lengths[i + 1] / lengths[i]) <= 0.7
    ]
    print(f"Debug: Found {len(division_frames)} division points in sequence of length {len(lengths)}")

    # בונים את כל המחזורים בציר הזמן: מתחילת הסדרה עד חלוקה ראשונה,
    # בין חלוקות עוקבות, ומחלוקה אחרונה עד סוף הסדרה.
    # לאחר מכן מסירים את המחזור הראשון והאחרון (חלקיים).
    cycle_starts = [0] + [idx + 1 for idx in division_frames]
    cycle_ends = division_frames + [len(lengths) - 1]
    candidate_cycles = list(zip(cycle_starts, cycle_ends))

    if len(candidate_cycles) <= 2:
        return cycles

    for cycle_num, (idx_x0, idx_xf) in enumerate(candidate_cycles[1:-1], start=1):
        
        # חילוץ כל נקודות האורך השייכות למחזור הנוכחי
        cycle_lengths = lengths[idx_x0 : idx_xf + 1]
        
        if len(cycle_lengths) < 2 or np.any(cycle_lengths <= 0):
            continue
            
        # יצירת וקטור זמן תואם למחזור (בדקות, החל מ-0)
        cycle_times = np.arange(len(cycle_lengths)) * TIME_STEP_MINUTES
        
        # ביצוע הפיכה לליניארית באמצעות לוגריתם טבעי (ln)
        log_lengths = np.log(cycle_lengths)
        
        # ביצוע Exponential Fit
        slope, intercept = np.polyfit(cycle_times, log_lengths, 1)
        
        x0_fitted = np.exp(intercept)
        alpha_fitted = slope
        actual_x0 = lengths[idx_x0]
        actual_xf = lengths[idx_xf]
        tau_min = (idx_xf - idx_x0) * TIME_STEP_MINUTES
        
        # חישוב המדדים
        phi = alpha_fitted * tau_min
        delta = actual_xf - actual_x0
        
        cycles.append(
            {
                'X0': actual_x0,
                'Xf': actual_xf,
                'tau_min': tau_min,
                'alpha': alpha_fitted,
                'Phi': phi,
                'Delta': delta,
            }
        )

        # if plt is not None:
        #     plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
        #     os.makedirs(plots_dir, exist_ok=True)
        #     plot_path = os.path.join(
        #         plots_dir,
        #         f"{filename_base}_{side_name}_cycle_{cycle_num}.png",
        #     )
        #     plt.figure()
        #     plt.plot(cycle_times, cycle_lengths, 'o', label='Data')
        #     plt.plot(cycle_times, np.exp(intercept + slope * cycle_times), '-', label='Fit')
        #     plt.xlabel('Time (minutes)')
        #     plt.ylabel('Length (microns)')
        #     plt.title(f'{filename_base} | {side_name} | Cycle {cycle_num}')
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig(plot_path)
        #     plt.close()
    return cycles

def process_analysis_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('_FINAL_corrected_data.csv'):
            print(f"מנתח: {filename}...")
            filename_base = filename.replace('_FINAL_corrected_data.csv', '')
            
            df = pd.read_csv(os.path.join(folder_path, filename))
            if 'Side' in df.columns:
                side_df = df[['Frame', 'Side', 'Length_Microns']].copy()
            else:
                side_df = pd.concat([
                    df[['Frame', 'Length Left']].rename(columns={'Length Left': 'Length_Microns'}).assign(Side='Left'),
                    df[['Frame', 'Length Right']].rename(columns={'Length Right': 'Length_Microns'}).assign(Side='Right'),
                ], ignore_index=True)
            side_df['Side'] = side_df['Side'].astype(str).str.strip().str.lower()
            
            output_file = os.path.join(folder_path, filename.replace('_FINAL_corrected_data', '_cycles.csv'))
            
            # שלב א': כתיבת המחזורים הבודדים לקובץ הזמני
            with open(output_file, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['Side', 'Cycle', 'X0', 'Xf', 'Alpha', 'Tau_Min', 'F_ratio', 'Phi', 'Delta', 'R'])
                
                for side in ['left', 'right']:
                    lengths = side_df[side_df['Side'] == side]['Length_Microns'].values
                    print(f"   צד {side.capitalize()}: נמצאו {len(lengths)} שורות בקובץ המקורי")
                    
                    cycles = analyze_cycles_robust(lengths, side, filename_base)
                    
                    prev_xf = None
                    for i, c in enumerate(cycles):
                        f_ratio = c['X0'] / prev_xf if prev_xf else None
                        R_val = f_ratio * np.exp(c['alpha'] * c['tau_min']) if f_ratio is not None else None
                        
                        w.writerow([
                            side.capitalize(), 
                            i+1, 
                            c['X0'], 
                            c['Xf'], 
                            c['alpha'], 
                            c['tau_min'], 
                            f_ratio, 
                            c['Phi'], 
                            c['Delta'], 
                            R_val
                        ])
                        prev_xf = c['Xf']

            # --- שלב ב': חישוב והוספת שורות הסיכום והממוצעים לבקשתך ---
            # קוראים את מה שכתבנו הרגע לתוך דאטה-פריים כדי לעשות ממוצעים בקלות
            res_df = pd.read_csv(output_file)
            
            # רשימת העמודות שעליהן נבצע את חישובי הממוצעים
            calc_cols = ['X0', 'Xf', 'Alpha', 'Tau_Min', 'F_ratio', 'Phi', 'Delta', 'R']
            
            if not res_df.empty:
                # 1. ממוצע כללי (ימין ושמאל יחד)
                mean_all = res_df[calc_cols].mean()
                # 2-3. ממוצעים נפרדים לפי צד
                mean_left = res_df[res_df['Side'] == 'Left'][calc_cols].mean()
                mean_right = res_df[res_df['Side'] == 'Right'][calc_cols].mean()
                # 4-5. הממוצע של הצד פחות הממוצע הכללי
                diff_left = mean_left - mean_all
                diff_right = mean_right - mean_all
                
                # בניית שורות הממוצע החדשות
                summary_rows = [
                    {'Side': 'SUMMARY', 'Cycle': 'Grand_Mean', **mean_all.to_dict()},
                    {'Side': 'SUMMARY', 'Cycle': 'Left_Mean', **mean_left.to_dict()},
                    {'Side': 'SUMMARY', 'Cycle': 'Right_Mean', **mean_right.to_dict()},
                    {'Side': 'SUMMARY', 'Cycle': 'Left_Minus_Grand', **diff_left.to_dict()},
                    {'Side': 'SUMMARY', 'Cycle': 'Right_Minus_Grand', **diff_right.to_dict()}
                ]
                
                # הפיכת רשימת השורות ל-DataFrame וחיבורן לתחתית הקובץ
                summary_df = pd.DataFrame(summary_rows)
                final_df = pd.concat([res_df, summary_df], ignore_index=True)
                
                # שמירה מחדש של הקובץ המלא עם הממוצעים
                final_df.to_csv(output_file, index=False)
                
            print(f"נשמר עם ממוצעים: {output_file}\n")

if __name__ == "__main__":
    from tkinter import filedialog

    analysis_dir = filedialog.askdirectory(title="analysis Folder")
  
    if analysis_dir:
        process_analysis_folder(analysis_dir)
