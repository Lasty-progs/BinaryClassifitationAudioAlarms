import os, shutil
import pandas as pd


CLASSIFICATION_CLASS = "Alarm" # Chose one class from vocabulary.csv

def convertFlac(files:list, directory:str) -> None:
    '''files must be name and extention, like "123.flac" \n
    directory must be "eval" or "dev"'''

    
    save_dir = "./temp/wavs/" + directory + "/"
    directory = "./FSD50K/" + directory + "_audio/"


    for file in files:    
        os.system('ffmpeg -i ' + directory + file +' '+ save_dir + file.split(".")[0] + '.wav')


# files = os.listdir(directory)

def getFileNames():
    files = {"dev":os.listdir("./FSD50K/" + "dev" + "_audio/"),
             "eval":os.listdir("./FSD50K/" + "eval" + "_audio/")}
    return files

files = getFileNames()

# print(os.listdir("./"))
# out_file = []
# with open("fsd50K/fsd50K_eval.tsv") as file:
#     tsv_file = csv.reader(file, delimiter="\t")

#     for line in tsv_file:
#         out_file.append([line[1][:len(line[1])-4] + "wav",
#                          "4" in (line[2].split(','))])
        
# with open("eval.txt", "w") as f:
#     for line in out_file:
#         f.write(line[0] + ',' + str(1 if line[1] else 0))
#         f.write('\n')