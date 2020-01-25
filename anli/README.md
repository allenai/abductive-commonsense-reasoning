
# Finetune bert model for ANLI
```
python anli/run_anli.py --task_name anli --model_name_or_path bert-large-uncased --batch_size 8 --lr 1e-5 --epochs 4 --output_dir models/bert-ft-lr1e-5-batch8-epoch4/ --data_dir data/anli/ --finetuning_model BertForMultipleChoice --max_seq_length 68 --tb_dir models/bert-ft-lr1e-5-batch8-epoch4/tb/ --warmup_proportion 0.2 --training_data_fraction 1.0 --seed 21004 --metrics_out_file models/bert-ft-lr1e-5-batch8-epoch4/metrics.json
```

# Compute max seq length for a model
```
python anli/max_ctx_for_dataset.py --data_dir data/anli/ --bert_model bert-base-uncased
```