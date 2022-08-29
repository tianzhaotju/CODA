# CODA
### Code Difference Guided Attacking for Deep Code Models
<img src="./figs/overview.png" alt="drawing" width="800">

--- ---

## Docker
Our experiments were conducted under Ubuntu 20.04. 
We have made a ready-to-use docker image for this experiment.
```shell
docker pull anonymous4open/coda:v1.3
```
Then, assuming you have NVIDIA GPUs, you can create a container using this docker image. 
An example:
```shell
docker run --name=coda --gpus all -it --mount type=bind,src=./coda,dst=/workspace anonymous4open/coda:v1.3
```

## Subjects
####  (1) Statistics of datasets and of target models.
<img src="./figs/statistics.png" alt="drawing" width="800">

--- --- ---

## Demo
Let's take the CodeBERT and Authorship Attribution task as an example. 
The `dataset` folder contains the training and evaluation data for this task. 
Run python attack.py in each directory to attack the deep code models.
E.g., run the following commands to attack the CodeBERT model on Authorship Attribution.

```shell
cd /root/Attack/CODA/AuthorshipAttribution/code/;
CUDA_VISIBLE_DEVICES=0 python attack.py --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt --model_name=codebert;
```


## Running Experiments
We refer to the README.md files under each folder to prepare the dataset and attack models on different tasks. 


## Experimental Results

####  (1) Comparison results of Attack Success Rate (ASR) on attacking CodeBERT and GraphCodeBERT across five tasks.
<img src="./figs/asr.png" alt="drawing" width="1000">

--- --- ---



####  (2) Comparison results of prediction confidence decrement (PCD) and invocations on attacking CodeBERT and GraphCodeBERT.
<img src="./figs/pcd_inv.png" alt="drawing" width="1000">

--- --- ---



####  (3) Experimental results of the user study to evaluate naturalness of adversarial examples generated by three techniques.
<img src="./figs/naturalness.png" alt="drawing" width="800">

--- --- ---



####  (4) Robustness improvement of the target models after adversarial fine-tuning.
<img src="./figs/retrain.png" alt="drawing" width="1000">

--- --- ---



####  (5) We investigated the contribution of each main component in CODA, including reference inputs selection (RIS), equivalent structure transformations (EST), and identifier renaming transformations (IRT).
<img src="./figs/ablation.png" alt="drawing" width="1000">

--- --- ---



#### (6) The influence of U in terms of average ASR across all the tasks.
<img src="./figs/U.png" alt="drawing" width="800">

--- ---




#### (7) The influence of N in terms of average ASR across all the tasks.
<img src="./figs/N.png" alt="drawing" width="1000">

--- ---


## Acknowledgement
We are very grateful that the authors of Tree-sitter, CodeBERT, GraphCodeBERT, ALERT, and CARROT make their code publicly available so that we can build this repository on top of their code. 
