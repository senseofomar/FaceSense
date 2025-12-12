import csv, os
def append_features_csv(row, folder="data", filename="features.csv"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs(os.path.join(project_root, folder), exist_ok=True)
    path = os.path.join(project_root, folder, filename)
    header = ["ts","expression","confidence","mouth_w","lip_gap","curve","brow","x1","y1","x2","y2"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    return os.path.abspath(path)
