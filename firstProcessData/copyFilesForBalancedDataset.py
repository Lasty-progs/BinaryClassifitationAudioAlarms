import csv
import shutil

filedir = "eval"
# filedir = "dev"
SIZE_NEGATIVES = 1280 if filedir == "dev" else 584

file_txt_out = []
with open("fsd50K/" + filedir + ".txt") as file:
    tsv_file = csv.reader(file, delimiter=",")
    count = 0

    for line in tsv_file:
        if line[1] == "1":
            file_txt_out.append(line)
            shutil.copy2(line[0], "dataset" + line[0][6:])
        elif count<= SIZE_NEGATIVES:
            count += 1
            file_txt_out.append(line)
            shutil.copy2(line[0], "dataset" + line[0][6:])

with open("dataset" + filedir + ".txt", "w") as f:
    for line in file_txt_out:
        f.write(line[0] + ',' + line[1])
        f.write('\n')
