import os
import glob
import json
import pandas as pd

def main():
    tasks = ["RTE","MRPC","STS-B","CoLA"]
    splits = [1, 5, 10, 100, 1000]
    replicates = range(5)
    ffn_options = ["ffn_hi_input", "ffn_ho_input"]

    for ffn_input in ffn_options:
        for task in tasks:
            for split in splits:
                split_str = "split"+str(split)
                for replicate in replicates:
                    rep_str = "r"+str(replicate)

                    training_name = "" + str(split_str) + "_" + str(rep_str) + "_" + str(ffn_input) + "_" + str(task) + ""
                    output_dir = "checkpoints/glue/finetuning_script/" + str(training_name)
                    working_dir = "/home/jovyan/joe-cls-vol/class_projects/nlp_11711_project/efficient-tuning-code"

                    train_data_dir = "low_resource_glue_data/" + str(split_str) + "/" + str(rep_str) + "/" + str(task) + "/train.csv"
                    test_data_dir = "low_resource_glue_data/" + str(split_str) + "/" + str(rep_str) + "/" + str(task) + "/test.csv"
                    val_data_dir = "low_resource_glue_data/" + str(split_str) + "/" + str(rep_str) + "/" + str(task) + "/dev.csv"

                    bash_str = "python -u examples/pytorch/text-classification/run_glue.py "\
                    + "--model_name_or_path bert-base-uncased "\
                    + "--train_file " + str(train_data_dir) + " "\
                    + "--test_file " + str(test_data_dir) + " "\
                    + "--validation_file " + str(val_data_dir) + " "\
                    + "--do_train "\
                    + "--do_eval "\
                    + "--do_predict "\
                    + "--max_seq_length 128 "\
                    + "--per_device_train_batch_size 32 "\
                    + "--per_device_eval_batch_size 32 "\
                    + "--max_tokens_per_batch 0 "\
                    + "--adam_beta1 0.9 "\
                    + "--adam_beta2 0.98 "\
                    + "--adam_epsilon 1e-6 "\
                    + "--attn_mode lisa "\
                    + "--attn_option concat "\
                    + "--attn_gate none "\
                    + "--ffn_mode adapter "\
                    + "--ffn_option " + ffn_input + " "\
                    + "--ffn_gate none "\
                    + "--adapter_layernorm_option fixed_scalar "\
                    + "--adapter_init_option lora "\
                    + "--adapter_scalar 2 "\
                    + "--mh_reuse_proj true "\
                    + "--layer_norm_before 1 "\
                    + "--layer_norm_after 0 "\
                    + "--hi_lnbefore 1 "\
                    + "--mid_dim 800 "\
                    + "--preseqlen 16 "\
                    + "--ffn_bn_len 16 "\
                    + "--init_with_bert 1 "\
                    + "--seed " + str(replicate) + " "\
                    + "--unfreeze_params ef_ "\
                    + "--num_bias_layers 12 "\
                    + "--max_eval_samples 1600 "\
                    + "--gradient_accumulation_steps 1 "\
                    + "--max_steps -10 "\
                    + "--num_train_epochs 10 "\
                    + "--learning_rate 1e-4 "\
                    + "--lr_scheduler_type polynomial "\
                    + "--max_grad_norm 1 "\
                    + "--weight_decay 0.1 "\
                    + "--warmup_steps 0 "\
                    + "--warmup_ratio 0.06 "\
                    + "--max_seq_length 512 "\
                    + "--fp16 "\
                    + "--logging_steps 50 "\
                    + "--save_total_limit 1 "\
                    + "--evaluation_strategy epoch "\
                    + "--save_strategy epoch "\
                    + "--save_steps 5000 "\
                    + "--eval_steps 5000 "\
                    + "--load_best_model_at_end "\
                    + "--report_to wandb "\
                    + "--run_name glue.finetuning_script." + str(training_name) + " "\
                    + "--overwrite_output_dir "\
                    + "--disable_tqdm true "\
                    + "--metric_for_best_model loss "\
                    + "--greater_is_better true "\
                    + "--ddp_find_unused_parameter false "\
                    + "--output_dir " + str(output_dir) + " "\
                    + "2>&1 | tee " + str(output_dir) + "/log.txt"

                    # change directory and run bash command
                    os.chdir(working_dir)
                    os.system(bash_str)

                    # load up dataframes
                    if os.path.exists("low_resource_glue_data/efficient_finetuning_results.csv"):
                        results_df = pd.read_csv("low_resource_glue_data/efficient_finetuning_results.csv", index_col=[0])
                    else:
                        results_df = pd.DataFrame([])

                    if os.path.exists(output_dir+"/predict_results_None.txt") and os.path.exists(output_dir+"/all_results.json"):
                        predict_df = pd.read_csv(output_dir+"/predict_results_None.txt", sep="\t")
                        test_df = pd.read_csv(test_data_dir, sep=",")

                        # load json of eval results
                        with open(output_dir+"/all_results.json", "r") as f:
                            eval_results_dict = json.load(f)

                        # compare test results to test set
                        if task == "CoLA":
                            label = "label" # 2nd column
                        elif task == "MRPC":
                            label = "Quality"
                        elif task == "RTE":
                            label = "label"
                        elif task == "STS-B":
                            label = "score"
                        num_diff = test_df[label].compare(predict_df["prediction"]).shape[0]
                        test_acc = 1 - (num_diff/predict_df.shape[0])

                        # create results dataframe entry
                        df = pd.DataFrame(
                            [{
                                "test_acc": test_acc,
                                "task": task,
                                "split": split,
                                "replicate": replicate,
                                "ffn": ffn_input,
                            }]
                        )

                        # combine results into one dataframe
                        results_df = results_df.append(df, ignore_index=True)

                    results_df.to_csv("low_resource_glue_data/efficient_finetuning_results.csv")

                    if os.path.exists(output_dir+"/pytorch_model.bin"):
                        os.remove(output_dir+"/pytorch_model.bin")
                        checkpoint_names = glob.glob(output_dir+"/checkpoint*")
                        for check_name in checkpoint_names:
                            if os.path.exists(check_name+"/optimizer.pt") and os.path.exists(check_name+"/pytorch_model.bin"):
                                os.remove(check_name+"/optimizer.pt")
                                os.remove(check_name+"/pytorch_model.bin")

    i = 0
    rename_path = "low_resource_glue_data/efficient_finetuning_results" + str(i) + ".csv"
    while os.path.exists(rename_path):
        i = i+1
        rename_path = "low_resource_glue_data/efficient_finetuning_results" + str(i) + ".csv"
    os.rename("low_resource_glue_data/efficient_finetuning_results.csv", rename_path)
                    

if __name__ == "__main__":
    main()
