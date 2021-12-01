import os
import pandas as pd
import csv

split_paths = [
    "/home/jovyan/working/class_projects/nlp_11711_project/revisit-bert-finetuning/low_resource/low_resource_glue_data/split1",
    "/home/jovyan/working/class_projects/nlp_11711_project/revisit-bert-finetuning/low_resource/low_resource_glue_data/split5",
    "/home/jovyan/working/class_projects/nlp_11711_project/revisit-bert-finetuning/low_resource/low_resource_glue_data/split10",
    "/home/jovyan/working/class_projects/nlp_11711_project/revisit-bert-finetuning/low_resource/low_resource_glue_data/split100",
    "/home/jovyan/working/class_projects/nlp_11711_project/revisit-bert-finetuning/low_resource/low_resource_glue_data/split1000",
]

for split_path in split_paths:
    split = int(split_path.split("t")[-1])
    for root, dirs, files in os.walk(split_path):
        if len(files) > 0:
            for file in files:
                if file.split(".")[-1] == "tsv":
                    read_header = "infer"
                    to_header = True
                    if "CoLA" in root:
                        read_header = None
                        to_header = False
                    print("reading "+root+"/"+file)
                    tsv_frame = pd.read_csv(root+"/"+file, sep="\t", error_bad_lines=False, header=read_header, engine="python", quoting=csv.QUOTE_NONE)

                    params = root.split("/")
                    to_dir = ""

                    to_dir = to_dir + params[-3] + "/"
                    if not os.path.isdir(to_dir):
                        os.mkdir(to_dir)

                    to_dir = to_dir + params[-2] + "/"
                    if not os.path.isdir(to_dir):
                        os.mkdir(to_dir)

                    to_dir = to_dir + params[-1] + "/"
                    if not os.path.isdir(to_dir):
                        os.mkdir(to_dir)
                    print("loaded " + to_dir + file + " creating " + to_dir+file.split(".")[0]+".csv")

                    tsv_frame.to_csv(to_dir+file.split(".")[0]+".csv", sep=",", index=False, header=to_header)

