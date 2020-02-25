# abductive-commonsense-reasoning
Public repository associated with [Abductive Commonsense Reasoning, ICLR 2020](https://arxiv.org/abs/1908.05739)


# Setup

1. Clone the repo
    ```
    git clone git@github.com:allenai/abductive-commonsense-reasoning.git
    ```
2. Install requirements
    ```
   pip install -r requirements.txt
   ```
3. Download Data
    ```
    sh get-data.sh
    ```

# Download Trained Model
```
wget https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/models/anli/bert-ft-lr1e-5-batch8-epoch4.tar.gz

mkdir -p models

tar -xvzf bert-ft-lr1e-5-batch8-epoch4.tar.gz -C models/
```

# Interactive Demo
```
python anli/demo.py --saved_model_dir models/bert-ft-lr1e-5-batch8-epoch4/ --gpu_id 0 --interactive
```

# Tasks
1. [Abductive Inference](anli/README.md)
2. [Abductive Generation](anlg/README.md)

# References
```
@inproceedings{
bhagavatula2020abductive,
title={Abductive Commonsense Reasoning},
author={Chandra Bhagavatula and Ronan Le Bras and Chaitanya Malaviya and Keisuke Sakaguchi and Ari Holtzman and Hannah Rashkin and Doug Downey and Wen-tau Yih and Yejin Choi},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Byg1v1HKDB}
}
```