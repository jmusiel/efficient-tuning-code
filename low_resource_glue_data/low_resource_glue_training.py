import os


def main():
    tasks = ["RTE","MRPC","STS-B","CoLA"]
    splits = [1, 5, 10, 100, 1000]
    replicates = range(5)

    tasks = ["RTE"]
    splits = [1]
    replicates = range(1)

    for task in tasks:
            for split in splits:
                split_str = "split"+str(split)
                for replicate in replicates:
                    rep_str = "r"+str(replicate)

                    training_name = "" + str(split_str) + "_" + str(rep_str) + "_" + str(task) + ""

                    bash_str = ""\
                    + "--model_name_or_path roberta-base "\
                    + "--train_file /home/jovyan/working/class_projects/nlp_11711_project/efficient-tuning-code/low_resource_glue_data/" + str(split_str) + "/" + str(rep_str) + "/" + str(task) + "/train.csv "\
                    + "--test_file /home/jovyan/working/class_projects/nlp_11711_project/efficient-tuning-code/low_resource_glue_data/" + str(split_str) + "/" + str(rep_str) + "/" + str(task) + "/test.csv "\
                    + "--validation_file /home/jovyan/working/class_projects/nlp_11711_project/efficient-tuning-code/low_resource_glue_data/" + str(split_str) + "/" + str(rep_str) + "/" + str(task) + "/dev.csv "\
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
                    + "--ffn_option ffn_hi_input "\
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
                    + "--max_steps -1 "\
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
                    + "--save_total_limit 2 "\
                    + "--evaluation_strategy epoch "\
                    + "--save_strategy epoch "\
                    + "--save_steps 5000 "\
                    + "--eval_steps 5000 "\
                    + "--load_best_model_at_end "\
                    + "--report_to wandb "\
                    + "--run_name rte.finetuning_script." + str(training_name) + " "\
                    + "--overwrite_output_dir "\
                    + "--disable_tqdm true "\
                    + "--metric_for_best_model loss "\
                    + "--greater_is_better true "\
                    + "--ddp_find_unused_parameter false "\
                    + "--output_dir checkpoints/glue/rte/finetuning_script/" + str(training_name) + " "\
                    + "2>&1 | tee checkpoints/glue/rte/finetuning_script/" + str(training_name) + "/log.txt"

                    os.chdir("/home/jovyan/working/class_projects/nlp_11711_project/efficient-tuning-code")
                    os.system(bash_str)


if __name__ == "__main__":
    main()
