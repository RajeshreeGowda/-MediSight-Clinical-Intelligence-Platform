import os

os.makedirs("discharge_notes", exist_ok=True)

# All text is pure ASCII - no special characters, no dashes, no smart quotes
notes = [
    ("Patient_001_Diabetes.txt",
     "Patient John D. Age 67. Male. Insurance Medicare. "
     "Diagnosis Diabetic Ketoacidosis. Blood glucose 480 mg per dL. "
     "HbA1c 11.2 percent. IV insulin drip started. "
     "Comorbidities Hypertension CKD Obesity. LOS 6 days. "
     "Discharge Home. Medications Metformin Lisinopril Insulin. "
     "Readmitted within 30 days YES."),

    ("Patient_002_HeartFailure.txt",
     "Patient Mary S. Age 72. Female. Insurance Medicare. "
     "Diagnosis Acute Decompensated Heart Failure. EF 30 percent. "
     "BNP 1800. Furosemide IV. LOS 9 days. Discharge SNF. "
     "Medications Furosemide Carvedilol Spironolactone. "
     "Low sodium diet. Daily weights. "
     "Readmitted within 30 days YES."),

    ("Patient_003_Pneumonia.txt",
     "Patient Robert K. Age 55. Male. Insurance Private. "
     "Diagnosis Community Acquired Pneumonia. O2 sat 88 percent. "
     "CXR RLL infiltrate. LOS 4 days. Discharge Home. "
     "Antibiotics Ceftriaxone Azithromycin. "
     "Readmitted within 30 days NO."),

    ("Patient_004_COPD.txt",
     "Patient Susan T. Age 68. Female. Insurance Medicaid. "
     "Diagnosis COPD Exacerbation. FEV1 42 percent. "
     "SpO2 84 percent on room air. LOS 8 days. "
     "Discharge Home Health Care. "
     "Medications Tiotropium Prednisone Azithromycin. "
     "Home oxygen 2L per min. "
     "Readmitted within 30 days YES."),

    ("Patient_005_Sepsis.txt",
     "Patient James L. Age 78. Male. Insurance Medicare. "
     "Diagnosis Sepsis from UTI. WBC 22k Lactate 3.2 Ecoli bacteremia. "
     "ICU stay 3 days. LOS 14 days. Discharge Rehab Facility. "
     "Antibiotics Piperacillin Tazobactam. "
     "Readmitted within 30 days YES."),

    ("Patient_006_HipFracture.txt",
     "Patient Elizabeth M. Age 83. Female. Insurance Medicare. "
     "Diagnosis Right Hip Fracture after fall. ORIF surgery. "
     "PT OT started day 2. LOS 7 days. Discharge SNF. "
     "Medications Enoxaparin Acetaminophen Calcium Vitamin D. "
     "Readmitted within 30 days NO."),

    ("Patient_007_Stroke.txt",
     "Patient David P. Age 70. Male. Insurance Medicare. "
     "Diagnosis Ischemic Stroke Left MCA. tPA given within 3 hours. "
     "LOS 6 days. Discharge Inpatient Rehab. "
     "Medications Aspirin Atorvastatin Lisinopril. "
     "Speech therapy and PT started. "
     "Readmitted within 30 days NO."),

    ("Patient_008_Appendectomy.txt",
     "Patient Priya R. Age 28. Female. Insurance Private. "
     "Diagnosis Acute Appendicitis. Laparoscopic surgery uncomplicated. "
     "LOS 1 day. Discharge Home. "
     "Medications Ibuprofen Ondansetron. "
     "Readmitted within 30 days NO."),

    ("Patient_009_RenalFailure.txt",
     "Patient Ravi S. Age 62. Male. Insurance Medicare. "
     "Diagnosis ESRD missed dialysis fluid overload. "
     "Potassium 6.8 mEq per L. Haemodialysis twice performed. "
     "LOS 5 days. Discharge Home with 3 times per week dialysis. "
     "Medications Sevelamer Epoetin Cinacalcet. "
     "Readmitted within 30 days YES."),

    ("Patient_010_ChestPain.txt",
     "Patient Anita K. Age 45. Female. Insurance Private. "
     "Diagnosis Chest Pain rule out ACS. Troponins negative x3. "
     "Stress test normal. LOS 1 day. Discharge Home. "
     "Medications Aspirin Atorvastatin. "
     "Readmitted within 30 days NO."),
]

print("Creating discharge notes...")
for filename, content in notes:
    filepath = os.path.join("discharge_notes", filename)
    # Write as ASCII only - eliminates all encoding errors
    safe_content = content.encode("ascii", errors="ignore").decode("ascii")
    with open(filepath, "w", encoding="ascii") as f:
        f.write(safe_content)
    print("  OK  " + filename)

print()
print("Done - " + str(len(notes)) + " notes created in discharge_notes folder")
print("Next step: python rag_pipeline.py")