# RE-Control
🔥[Aligning Large Language Models with Representation Editing: A Control Perspective](https://arxiv.org/abs/2406.05954)

RE-Control aligns LLMs by introducing external control signals into the hidden states of a pre-trained LLM during test time. 

![image](overview.jpg)



There are two environments for this project. For all programs except metrics.py you can use the environment llm.txt. For metrics.py, you can use the environment metric.txt.

## Installation (RE-Control)

Clone project and create environment with conda:
```
conda create -n recontrol python==3.10
conda activate recontrol

pip install -r llm.txt
```

**Note**: you may need to adjust the torch (cuda) version according to your GPU.

## Training process

First, we need to get the activations from the LLM:

`python get_activations_only.py --model_name llama3_8B --dataset_name shp`

Then, we need to label the activations with a reward model:

`python reward_label.py --model_name llama3_8B --dataset_name shp --reward_model openbmb/UltraRM-13b --mode train`

Train a value model:  
`python train_value_model.py --model_name llama3_8B --dataset_name shp --lr 0.001`

Conduct intervened inference:  
`python inference_intervention.py --model_name llama3_8B --dataset_name shp --use_intervention True --lr 1.0 --epochs 30 --value_lr 0.001`

## Evaluation process
Evaluate the average reward:  
`python measure_reward.py --out_file llama3_8B_shp_0.001_30_1.0 --model_name llama3_8B --dataset_name shp --reward_model openbmb/UltraRM-13b`

Evaluate the diversity and coherence:  
`python metrics.py --run_name llama3_8B_shp_0.001_30_1.0`

Evaluate the GPT-4 win rate:  
`python gpt4_eval.py --run_name_red llama3_8B_shp_0.0001_30_1.0 --run_name_blue dataset/dataset_prefer`

You need to provide the preferred response in the dataset as 'run_name_blue'. We provide an exmaple in dataset_prefer.json.

## Citation
If you find our work helpful, please consider citing our paper:

```
@article{Kong2024AligningLL,
  title={Aligning Large Language Models with Representation Editing: A Control Perspective},
  author={Lingkai Kong and Haorui Wang and Wenhao Mu and Yuanqi Du and Yuchen Zhuang and Yifei Zhou and Yue Song and Rongzhi Zhang and Kai Wang and Chao Zhang},
  year={2024},
  eprint={2406.05954},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```
