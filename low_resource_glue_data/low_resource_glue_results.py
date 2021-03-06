import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import scipy

def get_test_metrics():
    test_metrics = {
        "RTE": (["test_acc"], 0),
        "MRPC": (["test_acc", "test_f1", "test_acc_and_f1"], 1),
        "STS-B": (["test_pearson", "test_spearmanr", "test_corr"], 1),
        "CoLA": (["test_mcc"], 0),
    }
    return test_metrics

def main():
    tasks = ["RTE","MRPC","STS-B","CoLA"]
    splits = [1, 5, 10, 100, 1000]
    replicates = range(5)
    ffn_options = ["ffn_hi_input", "ffn_ho_input"]
    results_df = pd.read_csv("/home/jovyan/joe-cls-vol/class_projects/nlp_11711_project/efficient-tuning-code/low_resource_glue_data/efficient_finetuning_results0.csv", index_col=[0])
    plot_values = {}

    for ffn_input in ffn_options:
        plot_values[ffn_input] = {}
        for task in tasks:
            plot_values[ffn_input][task] = {
                "split": [],
                "mean": [],
                "std": [],
            }
            for split in splits:
                if task == "RTE":
                    split_results = results_df[(results_df["task"] == task) & (results_df["ffn"] == ffn_input) & (results_df["split"] == split)]["test_acc"]
                    mean_split = np.mean(split_results.to_numpy())
                    std_split = np.std(split_results.to_numpy())
                else:
                    actual_list = []
                    predicted_list = []
                    for replicate in replicates:
                        results_path = "/home/jovyan/joe-cls-vol/class_projects/nlp_11711_project/efficient-tuning-code/checkpoints/glue/finetuning_script/" +\
                            "split" + str(split) +\
                            "_r" + str(replicate) +\
                            "_" + str(ffn_input) +\
                            "_" + str(task) +\
                            "/predict_results_None.txt"
                        rep_df = pd.read_csv(results_path, sep="\t")
                        predicted_list.append(rep_df["prediction"].to_numpy())
                        test_data_path = "/home/jovyan/joe-cls-vol/class_projects/nlp_11711_project/efficient-tuning-code/low_resource_glue_data" +\
                            "/split" + str(split) +\
                            "/r" + str(replicate) +\
                            "/" + str(task) +\
                            "/test.csv"
                        test_data_df = pd.read_csv(test_data_path)
                        if task == "MRPC":
                            actual_list.append(test_data_df["Quality"].to_numpy())
                        elif task == "STS-B":
                            actual_list.append(test_data_df["score"].to_numpy())
                        elif task == "CoLA":
                            actual_list.append(test_data_df["label"].to_numpy())
                    score_list = []
                    for actual, pred in zip(actual_list, predicted_list):
                        if task == "MRPC":
                            score_list.append(sklearn.metrics.f1_score(actual, pred))
                        elif task == "STS-B":
                            score_list.append(scipy.stats.spearmanr(actual, pred)[0])
                        elif task == "CoLA":
                            score_list.append(sklearn.metrics.matthews_corrcoef(actual, pred))
                    mean_split = np.mean(score_list)
                    std_split = np.std(score_list)

                plot_values[ffn_input][task]["split"].append(split)
                plot_values[ffn_input][task]["mean"].append(mean_split)
                plot_values[ffn_input][task]["std"].append(std_split)

    # basically copy paste of plotting script

    for ffn_input in ffn_options:
        fig, axs = plt.subplots(1, 4, figsize=(20, 4))
        for dataset, ax in zip(tasks, axs):
            test_metrics = get_test_metrics()
            test_metric = test_metrics[dataset]
            test_metric = test_metric[0][test_metric[1]]

            ax.set_title(dataset, size=20)
            ax.set_ylabel(test_metric, size=20)
            ax.set_xlabel("Size of Split", size=20)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)

            # if dataset == "RTE":
            #     ylimits = [0.5, 0.75]
            # elif dataset == "MRPC":
            #     ylimits = [0.8, 0.95]
            # elif dataset == "STS-B":
            #     ylimits = [0.85, 0.9]
            # elif dataset == "CoLA":
            #     ylimits = [0.4, 0.7]
            ylimits = [0, 1]
            ax.set_ylim(ylimits)

            label = "Correction"
            color = "cornflowerblue"
            fillcolor = "lightsteelblue"

            x = np.array(plot_values[ffn_input][dataset]["split"])
            y = np.array(plot_values[ffn_input][dataset]["mean"])
            std = np.array(plot_values[ffn_input][dataset]["std"])
            ax.semilogx(x, y, color=color)
            ax.fill_between(x, y - std, y + std, color=fillcolor, alpha=0.5)
        fig.tight_layout()
        fig.suptitle(ffn_input, fontsize=20)
        plt.savefig("scored_figure_" + str(ffn_input) + "_lowresource_flat.png")


if __name__ == "__main__":
    main()
