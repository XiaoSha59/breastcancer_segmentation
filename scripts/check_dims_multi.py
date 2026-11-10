import glob
import gzip
import os
import struct

patients = ["ACRIN-6698-103939", "ACRIN-6698-104268", "ACRIN-6698-107700", "ACRIN-6698-108969"]

for pid in patients:
    files = sorted(glob.glob(f"/home/XiaoSha/data/raw/{pid}/**/*.nii.gz", recursive=True))
    if not files:
        continue
    f = files[0]
    with gzip.open(f, "rb") as gz:
        hdr = gz.read(348)
        dim = struct.unpack("<8h", hdr[40:56])
        ndim = dim[0]
        shape = dim[1 : ndim + 1]
        pixdim = struct.unpack("<8f", hdr[76:108])
        spacing = [round(pixdim[i], 3) for i in range(1, ndim + 1)]
        print(f"Patient {pid:18s} | Shape: {str(shape):16s} | Spacing: {spacing}")
