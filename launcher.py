import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
import threading
import os
import traceback

# ייבוא השלבים
from step1_detection import process_folder_automation, build_output_folder_from_input
from step2_manual_correction_qt import run_correction_qt, run_correction_qt_standalone
from step3_recalculate import process_recalculation_folder  
from step4_analysis import process_analysis_folder
from step5_summary import process_cycles_summary

# הגדרת מראה
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class CellAnalysisSuite(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.configure(fg_color="#264653")
        self.title("Sister Machine Analysis Tool")
        self.geometry("500x600") 
        
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "sister_machine_logo.ico")
            self.iconbitmap(icon_path)
        except:
            pass
            
        self.title_label = ctk.CTkLabel(self, text="Sister Machine Analysis", 
                                        text_color="#FFFFFF", font=("Roboto", 24, "bold"))
        self.title_label.pack(pady=35)
        
        btn_params = {
            "width": 350,
            "height": 50, 
            "corner_radius": 15,
            "text_color": "#264653"
        }
        
        self.btn1 = ctk.CTkButton(self, text="Step 1 - Cell Detection", 
                                  fg_color="#B6F8F7", border_width=3, border_color="#1BF0EC", 
                                  hover_color="#1BF0EC", **btn_params, command=self.run_step1)
        self.btn1.pack(pady=10)
        
        self.btn2 = ctk.CTkButton(self, text="Step 2 - Manual Correction", 
                                  fg_color="#82cfc6", border_width=3, border_color="#289c8f", 
                                  hover_color="#289c8f", **btn_params, command=self.run_step2)
        self.btn2.pack(pady=10)
        
        self.btn1b = ctk.CTkButton(self, text="Step 3 - Calculation for additional fluorophore", 
                       fg_color="#eed188", border_width=3, border_color="#cca030", 
                       hover_color="#cca030", **btn_params, command=self.run_step3)
        self.btn1b.pack(pady=10)

        self.btn3 = ctk.CTkButton(self, text="Step 4 - Data Analysis", 
                                  fg_color="#f5b480", border_width=3, border_color="#e27d2a", 
                                  hover_color="#e27d2a", **btn_params, command=self.run_step4)
        self.btn3.pack(pady=10)
        
        self.btn4 = ctk.CTkButton(self, text="Step 5 - Summary", 
                                  fg_color="#f08d74", border_width=3, border_color="#e0451f", 
                                  hover_color="#e0451f", **btn_params, command=self.run_step5)
        self.btn4.pack(pady=10)
        
        self.status_var = ctk.StringVar(value="Status: Ready")
        self.status_label = ctk.CTkLabel(self, textvariable=self.status_var, text_color="#FFFFFF")
        self.status_label.pack(side="bottom", pady=20)
        
    def set_status(self, message):
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Status: {message}")
        self.update_idletasks()
        
    def disable_buttons(self):
        self.btn1.configure(state="disabled")
        self.btn1b.configure(state="disabled")
        self.btn2.configure(state="disabled")
        self.btn3.configure(state="disabled")
        self.btn4.configure(state="disabled")
        
    def enable_buttons(self):
        self.btn1.configure(state="normal")
        self.btn1b.configure(state="normal")
        self.btn2.configure(state="normal")
        self.btn3.configure(state="normal")
        self.btn4.configure(state="normal")
        
    def run_step1(self):
        input_dir = filedialog.askdirectory(title="Select TIF Input Folder")
        if not input_dir: return
        output_dir = build_output_folder_from_input(input_dir)
        
        def worker():
            try:
                self.disable_buttons()
                def ui_callback(msg):
                    self.status_var.set(msg)
                    self.update_idletasks()
                process_folder_automation(input_dir, output_dir, callback=ui_callback)
                messagebox.showinfo("Success", f"Completed!\nOutput: {output_dir}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                self.enable_buttons()
                self.status_var.set("Status: Ready")
                
        threading.Thread(target=worker, daemon=True).start()

    def run_step3(self):
        npy_folder = filedialog.askdirectory(title="Select Folder With .npy Segmentation Files")
        if not npy_folder: return
        
        tif_folder = filedialog.askdirectory(title="Select Folder With New Fluorophore .tif Files")
        if not tif_folder: return
        
        output_suffix = simpledialog.askstring(
            "Output File Suffix",
            "Enter a suffix to add to the output file names:\n"
            "Example: GFP or experiment_2",
            parent=self,
        )
        if output_suffix is None:
            return

        output_suffix = output_suffix.strip()
        if not output_suffix:
            messagebox.showwarning("Missing Suffix", "Please enter a suffix for the output file names.")
            return

        invalid_characters = '<>:"/\\|?*'
        if any(character in output_suffix for character in invalid_characters):
            messagebox.showwarning(
                "Invalid Suffix",
                "The suffix cannot contain: < > : \" / \\ | ? *",
            )
            return

        
        def worker():
            try:
                self.disable_buttons()
                
                def ui_callback(msg):
                    self.status_var.set(f"Status: {msg}")
                    self.update_idletasks()
                
                # הרצת העיבוד המעודכן שמחזיר גם את רשימת הלא-משויכים
                matched, total, unmatched_tifs = process_recalculation_folder(
                    npy_folder, tif_folder, output_suffix, callback=ui_callback
                )
                
                # הודעת סיום רגילה
                messagebox.showinfo(
                    "Batch Complete", 
                    f"Successfully processed {matched} out of {total} TIF files.\nResults saved to:\n{npy_folder}"
                )
                
                # במידה ויש קבצים שלא שויכו, תקפוץ התרעה מפורשת
                if unmatched_tifs:
                    unmatched_list = "\n".join(unmatched_tifs[:15]) # הצגת עד 15 קבצים כדי לא להעמיס על המסך
                    if len(unmatched_tifs) > 15:
                        unmatched_list += f"\n...and {len(unmatched_tifs) - 15} more files."
                        
                    messagebox.showwarning(
                        "Missing Backup Files",
                        f"The following {len(unmatched_tifs)} TIF files were SKIPPED because no matching .npy file was found:\n\n{unmatched_list}"
                    )
                    
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Error", str(e))
            finally:
                self.enable_buttons()
                self.status_var.set("Status: Ready")
                
                threading.Thread(target=worker, daemon=True).start()

    def run_step2(self):
        try:
            self.disable_buttons()
            self.set_status("Running Manual Correction...")
            run_correction_qt_standalone()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))
        finally:
            self.enable_buttons()
            self.set_status("Ready")

    def run_step4(self):
        analysis_dir = filedialog.askdirectory(title="Select Analysis Folder")
        if not analysis_dir: return
        
        def worker():
            try:
                self.disable_buttons()
                self.set_status("Running Data Analysis...")
                process_analysis_folder(analysis_dir)
                self.set_status("Finished")
                messagebox.showinfo("Finished", "Analysis Completed")
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Error", str(e))
            finally:
                self.enable_buttons()
                self.set_status("Ready")
                
        threading.Thread(target=worker, daemon=True).start()

    def run_step5(self):
        analysis_dir = filedialog.askdirectory(title="Select Folder With _cycles.csv.csv Files")
        if not analysis_dir: return
        
        def worker():
            try:
                self.disable_buttons()
                self.set_status("Running Cycles Summary...")
                output_file = process_cycles_summary(analysis_dir)
                self.set_status("Finished")
                messagebox.showinfo("Finished", f"Cycles summary saved:\n{output_file}")
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Error", str(e))
            finally:
                self.enable_buttons()
                self.set_status("Ready")
                
        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():
        print(f" Success! GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print(" Warning: GPU NOT detected! Running on CPU mode.")
        
    app = CellAnalysisSuite()
    app.mainloop()
