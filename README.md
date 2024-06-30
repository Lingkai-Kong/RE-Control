# RE-Control
Aligning large language models (LLMs) with human objectives is crucial for real-world applications. However, fine-tuning LLMs for alignment often suffers from unstable training and requires substantial computing resources. Test-time alignment techniques, such as prompting and guided decoding, do not modify the underlying model, and their performance remains dependent on the original model's capabilities. To address these challenges, we propose aligning LLMs through representation editing. The core of our method is to view a pre-trained autoregressive LLM as a discrete-time stochastic dynamical system. To achieve alignment for specific objectives, we introduce external control signals into the state space of this language dynamical system. We train a value function directly on the hidden states according to the Bellman equation, enabling gradient-based optimization to obtain the optimal control signals at test time. Our experiments demonstrate that our method outperforms existing test-time alignment techniques while requiring significantly fewer resources compared to fine-tuning methods.

[Paper Link](https://arxiv.org/abs/2406.05954)

![image](overview.jpg)



There are two environments for this project. For all programs except metrics.py you can use the environment llm.txt. For metrics.py, you can use the environment metric.txt.

Prepare the activation dataset:

`python get_activations_only.py --model_name vicuna_7B --dataset_name Anthropic/hh-rlhf`

Label the activation with reward:

`python reward_label.py --mode train`

Train a value model:  
`python train_value_model.py vicuna hhrlhf --lr 0.0001 --device 0`

Conduct intervened inference:  
`python inference_intervention.py vicuna_7B --use_intervention True --lr 0.5 --epochs 30 --value_lr 0.0001 --device 2`

Evaluate the average reward:  
`python measure_reward_final.py --out_file vicuna`

Evaluate the diversity and coherence:  
`python metrics_final.py --run_name vicuna`

Evaluate the GPT-4 win rate:  
`python gpt4_eval.py --run_name_red vicuna`
