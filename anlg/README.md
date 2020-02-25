GPT2 finetuned with optional ways of integrating COMeT commonsense embeddings. 


  ### 1. `only_o1_o2`: 
  [Beaker Experiment](https://beaker.org/ex/ex_az1896fjqsn3/tasks/tk_68jhb1tkdec9)
  
    python anlg/run_generation.py --model_type gpt2_for_anli_comet --model_name_or_path models/anlg/only_o1_o2/gpt2/ --input-file data/anlg/dev-w-comet-preds.jsonl --task anli --output-file models/anlg/only_o1_o2/gpt2/dev-w-comet-preds-generations.jsonl

  ### 2. restricted_comet_phrases_prefix: 
  [Beaker Experiment](https://beaker.org/ex/ex_3dyhlvh0imww/tasks/tk_tosvsiv9usp8)

    python anlg/run_generation.py --model_type gpt2_for_anli_comet --model_name_or_path models/anlg/restricted_comet_phrases_prefix/gpt2/ --input-file data/anlg/dev-w-comet-preds.jsonl --task anli --output-file models/anlg/restricted_comet_phrases_prefix/dev-w-comet-preds-generations.jsonl --include_comet True --comet_as_text True --restrict_comet True

  ### 3. lm_all_comet_emb_prefix: 
  [Beaker Experiment](https://beaker.org/ex/ex_gyt7hofohynr/tasks/tk_i25t9wkovcwk)

    python anlg/run_generation.py --model_type gpt2_for_anli_comet --model_name_or_path models/anlg/lm_all_comet_emb_prefix/gpt2/ --input-file data/anlg/dev-w-comet-preds.jsonl --task anli --output-file models/anlg/lm_all_comet_emb_prefix/gpt2/dev-w-comet-preds-generations.jsonl --include_comet True --comet_model_path comet-model/atomic_pretrained_model.th

  ### 4. lm_restricted_comet_emb_prefix: 
  [Beaker Experiment](https://beaker.org/ex/ex_m0d33r2bucxb/tasks/tk_gl126vskq879)

  ### 5. all_comet_phrases_prefix: 
  [Beaker Experiment](https://beaker.org/ex/ex_1g2jh8ce2h1s/tasks/tk_wufqybmvxjxk)
   
    python anlg/run_generation.py --model_type gpt2_for_anli_comet --model_name_or_path models/anlg/all_comet_phrases_prefix/gpt2/ --input-file data/anlg/dev-w-comet-preds.jsonl --task anli --output-file models/anlg/all_comet_phrases_prefix/gpt2/dev-w-comet-preds-generations.jsonl --include_comet true --comet_as_text true