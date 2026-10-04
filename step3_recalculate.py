import os
import numpy as np
import tifffile as tiff
import csv
from skimage.measure import regionprops

# קבועים נדרשים
PIXEL_TO_MICRON = 1 / 16

def get_cell_metrics(mask):
    """מחשבת את המטריקות הגיאומטריות עבור התא מתוך המסכה שלו."""
    props = regionprops(mask.astype(int))
    if not props:
        return 0, 0, 1.0
    prop = props[0]
    length = prop.major_axis_length
    perimeter = prop.perimeter
    area = prop.area
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else:
        circularity = 0.0
    return length, perimeter, circularity

def process_single_file(fluor_image_path, backup_npy_path, output_csv_path):
    """מעבדת קובץ בודד (לוגיקת החישוב המקורית)."""
    backup_data = np.load(backup_npy_path, allow_pickle=True).item()
    
    with tiff.TiffFile(fluor_image_path) as tif:
        images = tif.asarray()
        if len(images.shape) == 2: 
            images = np.expand_dims(images, axis=0)
            
    all_frames_data = []
    
    for frame_idx, frame in enumerate(images):
        if frame_idx not in backup_data:
            continue
            
        chosen_cells = backup_data[frame_idx]["chosen"]
        frame_res = {"Frame": frame_idx + 1, "Left": None, "Right": None}
        
        for i, cell in enumerate(chosen_cells):
            side = "Left" if i == 0 else "Right"
            mask = cell["mask"]
            
            length_px, perimeter_px, circularity = get_cell_metrics(mask)
            mean_intensity = float(np.mean(frame[mask > 0])) if np.any(mask) else 0.0
            
            frame_res[side] = {
                "Length_Microns": length_px * PIXEL_TO_MICRON,
                "Perimeter_Microns": perimeter_px * PIXEL_TO_MICRON,
                "Circularity": circularity,
                "Area_Pixels": int(np.sum(mask)),
                "Mean_Intensity": mean_intensity,
                "Center_X": cell["cx"],
                "Center_Y": cell["cy"]
            }
        all_frames_data.append(frame_res)
        
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "Length Left", "Area Left", "Intensity Left", "Length Right", "Area Right", "Intensity Right"])
        for d in all_frames_data:
            left = d["Left"] or {}
            right = d["Right"] or {}
            w.writerow([
                d["Frame"],
                left.get("Length_Microns", 0),
                left.get("Area_Pixels", 0),
                left.get("Mean_Intensity", 0),
                right.get("Length_Microns", 0),
                right.get("Area_Pixels", 0),
                right.get("Mean_Intensity", 0),
            ])

def extract_file_id(filename):
    """מחלץ את המזהה הייחודי המשותף (למשל '10232025_Sis06') מתוך שם הקובץ."""
    parts = filename.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return filename

def process_recalculation_folder(npy_folder, tif_folder, output_suffix, callback=None):
    """סורקת תיקיות, משייכת קבצים לפי מזהה ייחודי משותף ומחזירה ספירה ורשימת קבצים שלא שויכו.
    כל קבצי ה-CSV נשמרים בתוך ה-npy_folder עם סיומת שנבחרה על ידי המשתמש."""
    if not os.path.exists(npy_folder) or not os.path.exists(tif_folder):
        raise FileNotFoundError("An input folder does not exist.")
        
    npy_files = [f for f in os.listdir(npy_folder) if f.lower().endswith(".npy")]
    tif_files = [f for f in os.listdir(tif_folder) if f.lower().endswith(".tif") or f.lower().endswith(".tiff")]
    
    total_files = len(tif_files)
    matched_count = 0
    unmatched_tifs = []
    
    if callback:
        callback(f"Scanning folders... Found {len(npy_files)} NPYs and {total_files} TIFs.")
        
    # בניית מילון עבור קבצי ה-NPY לפי המזהה הייחודי שלהם
    npy_dict = {}
    for npy_file in npy_files:
        file_id = extract_file_id(npy_file)
        npy_dict[file_id] = npy_file
        
    # מעבר על קבצי ה-TIF וביצוע הצימדוד
    for idx, tif_file in enumerate(tif_files):
        # שליפת המזהה הייחודי של קובץ ה-TIF הנוכחי
        tif_id = extract_file_id(tif_file)
        
        # בדיקה האם קיים קובץ NPY תואם באותו מזהה במילון
        if tif_id in npy_dict:
            matched_npy = npy_dict[tif_id]
            matched_count += 1
            
            if callback:
                callback(f"Processing File {idx + 1}/{total_files} | {tif_file}")
                
            full_tif_path = os.path.join(tif_folder, tif_file)
            full_npy_path = os.path.join(npy_folder, matched_npy)
            
            # יצירת נתיב שמירה המבוסס על שם ה-TIF בתוך תיקיית ה-NPY
            base_name, _ = os.path.splitext(tif_file)
            output_csv_name = f"{base_name}_{output_suffix}_recalculated_results.csv"
            full_output_path = os.path.join(npy_folder, output_csv_name)
            
            process_single_file(full_tif_path, full_npy_path, full_output_path)
        else:
            unmatched_tifs.append(tif_file)
            
    return matched_count, total_files, unmatched_tifs

