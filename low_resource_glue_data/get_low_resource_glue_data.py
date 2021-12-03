import os
import pandas as pd
import csv

base_split_path = "/home/jovyan/joe-cls-vol/class_projects/nlp_11711_project/revisit-bert-finetuning/low_resource/low_resource_glue_data/"

split_paths = [
    base_split_path+"split1",
    base_split_path+"split5",
    base_split_path+"split10",
    base_split_path+"split100",
    base_split_path+"split1000",
]

for split_path in split_paths:
    split = int(split_path.split("t")[-1])
    for root, dirs, files in os.walk(split_path):
        if len(files) > 0:
            for file in files:
                if file.split(".")[-1] == "tsv" and file != "test.tsv":
                    read_header = "infer"
                    to_header = True
                    if "CoLA" in root:
                        read_header = None
                        to_header = True
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

                    tsv_frame.replace(to_replace="2012test", value="2012", inplace=True)
                    tsv_frame.replace(to_replace="2012train", value="2012", inplace=True)
                    if "CoLA" in root:
                        tsv_frame.rename(columns={
                            0: "source",
                            1: "label",
                            2: "author",
                            3: "sentence",
                        }, inplace=True)

                    if file == "dev.tsv":
                        test_df = tsv_frame.sample(frac=0.5)
                        tsv_frame.drop(test_df.index)
                        test_df.to_csv(to_dir+"test"+".csv", sep=",", index=False, header=to_header)
                        print("created test split " + to_dir + "test.csv")

                    tsv_frame.to_csv(to_dir+file.split(".")[0]+".csv", sep=",", index=False, header=to_header)
                    print("created " + to_dir+file.split(".")[0]+".csv")

