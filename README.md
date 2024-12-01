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

`python get_activations_only.py --model_name vicuna_7B --dataset_name hh_rlhf`

Then, we need to label the activations with a reward model:

`python reward_label.py --model_name vicuna_7B --dataset_name hh_rlhf --reward_model argsearch/llama-7b-rm-float32 --mode train`

Train a value model:  
`python train_value_model.py --model_name vicuna_7B --dataset_name hh_rlhf --lr 0.0001 --device 0`

Conduct intervened inference:  
`python inference_intervention.py --model_name vicuna_7B --dataset_name hh_rlhf --use_intervention True --lr 0.5 --epochs 30 --value_lr 0.0001 --device 0`

## Evaluation process
Evaluate the average reward:  
`python measure_reward_final.py --out_file vicuna`

Evaluate the diversity and coherence:  
`python metrics_final.py --run_name vicuna`

Evaluate the GPT-4 win rate:  
`python gpt4_eval.py --run_name_red vicuna`


## Citation
If you find our work helpful, please consider citing our paper:

```
@article{Kong2024AligningLL,
  title={Aligning Large Language Models with Representation Editing: A Control Perspective},
  author={Lingkai Kong and Haorui Wang and Wenhao Mu and Yuanqi Du and Yuchen Zhuang and Yifei Zhou and Yue Song and Rongzhi Zhang and Kai Wang and Chao Zhang},
  journal={ArXiv},
  year={2024},
  volume={abs/2406.05954},
  url={https://api.semanticscholar.org/CorpusID:270372048}
}
```
