import glob
import os
import nibabel as nib

files = sorted(glob.glob("/home/XiaoSha/data/raw/ACRIN-6698-102212/**/*.nii.gz", recursive=True))
print(f"Total files in ACRIN-6698-102212: {len(files)}")
for f in files[:4]:
    img = nib.load(f)
    print("File:", os.path.basename(f))
    print("  Shape:", img.shape)
    print("  Spacing (zooms mm):", img.header.get_zooms())
    print("  Data type:", img.get_data_dtype())
