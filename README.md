# abductive-commonsense-reasoning
Public repository associated with [Abductive Commonsense Reasoning, ICLR 2020](https://arxiv.org/abs/1908.05739)


# Setup

1. Clone the repo
    ```
    git clone git@github.com:allenai/abductive-commonsense-reasoning.git
    ```
2. Install requirements in a Python 3.7 environment
    ```
   pip install -r requirements.txt
   ```
3. Download Data and Models
    ```
    sh get-data.sh
    ```
   
The script will create `data/` and `models/` directories containing subdirectories `anli` and `anlg` for the Abuctive Inference and Generation tasks respectively. The size of the data and models is around *12G*. If you'd like to download data:

    wget https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/anli.zip
    wget https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/anlg.zip
    


# Tasks
1. [Abductive Inference](anli/README.md)
    ### Interactive Demo
    ```
    python anli/demo.py --saved_model_dir models/bert-ft-lr1e-5-batch8-epoch4/ --gpu_id 0 --interactive
    ```

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
