import csv


# with open("dataset/eval.txt") as file:
#     tsv_file = csv.reader(file, delimiter=",")
#     count = 0
#     positives = 0

#     for line in tsv_file:
#         count += 1
#         if line[1] == "1":
#             positives += 1

# print("eval%" + str(positives/count))
# print(positives)

with open("temp/train_mels_150/dev.txt") as file:
    tsv_file = csv.reader(file, delimiter=",")
    count = 0
    positives = 0

    for line in tsv_file:
        count += 1
        if line[1] == "1":
            positives += 1

print("dev%" + str(positives/count))
print(positives)
