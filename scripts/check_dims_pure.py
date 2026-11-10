import glob
import gzip
import os
import struct

files = sorted(glob.glob("/home/XiaoSha/data/raw/ACRIN-6698-102212/**/*.nii.gz", recursive=True))
print(f"Total files in ACRIN-6698-102212: {len(files)}")

for f in files:
    with gzip.open(f, "rb") as gz:
        hdr = gz.read(348)
        # NIfTI-1 header: sizeof_hdr at offset 0 (int32)
        # dim at offset 40: short[8] -> dim[0]=ndim, dim[1..ndim]=dimensions
        dim = struct.unpack("<8h", hdr[40:56])
        ndim = dim[0]
        shape = dim[1 : ndim + 1]
        # pixdim at offset 76: float[8] -> pixdim[1..ndim]=voxel spacing
        pixdim = struct.unpack("<8f", hdr[76:108])
        spacing = [round(pixdim[i], 3) for i in range(1, ndim + 1)]

        fname = os.path.basename(f)
        print(f"File: {fname:45s} | Dim: {str(shape):18s} | Voxel Spacing (mm): {spacing}")
