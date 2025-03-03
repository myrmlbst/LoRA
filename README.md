# Fine-tuning AI Models with LoRA
> About the Webinar: Introduction to Fine-tuning focusing on Parameter Efficient Fine-tuning with Low-rank Adaptation (LoRA). LoRA significantly reduces the computational and storage requirements of traditional fine-tuning methods by decomposing weight updates into low-rank matrices.

This repository consists of notes I took from a webinar/workshop hosted by Ali Masri, PhD on March 3, 2025, in addition to the source code of the project used to demonstrate and apply the concepts in real life.
The workshop was about fine-tuning AI models with LoRa, a parameter-efficient fine-tuning technique that allows us to update the weights of a pre-trained model in a more efficient way by learning a low-rank approximation of the weight matrix.

## 1. Introduction to Fine-Tuning
### 1.1. Definition and Significance
Fine-tuning a model refers to the process of updating the weights of a pre-trained model on a new dataset. 
This is a common practice in deep learning, where the pre-trained model is used as a starting point to solve a new task or improve the performance on a specific dataset. 

Fine-tuning allows us to leverage the knowledge learned by the pre-trained model on a large dataset and adapt it to a new domain or task with a smaller dataset.

### 1.2. Appropriate Situations and Limitations
There are certain cases where expensive methods (in terms of money and time) that require a lot of data may be good to use (ex: limited resources).
- _When to fine-tune:_ 
  1. When you have a small dataset and the pre-trained model is trained on a similar task or domain.
- _When NOT to fine-tune:_ 
  1. During very generic tasks where the model already performs well... instead use prompt-engineering to help the model output better results
  2. When there are frequent updates, we use retrieval augmented generation (RAG) instead

## 2. Types of Fine-Tuning
Older techniques:
1. Feature Based Approach: Contains the pretrained transformer/neural network. The hidden layer is frozen, and only last layer is updated. This method is cheap but not very effective, because we're only changing the output's probability rather than addressing the root cause of the problem.
2. Fine-tuning 1: It is the same as Feature Based Approach, except for the classifier is also updated.
3. Fine-tuning 2: All layers are updated.

From a business perspective: OpenAI offers a fine-tuning service; if they were using the old-school methods, they would have to create a copy of the model and train a model from scratch for each customer, which would be impossible to make for multiple customers as each model takes up a lot of storage. However, with the new method, they can simply finetune the model for each customer according to their needs without having to create multiple copies of the original model beforehand.

A newer, more efficient technique: Backpropagation

## 3. Overview of Backpropagation in Neural Networks
A Neural Network consists of node layers divided into an input layer, a hidden layer, and an output layer.
Each connection between the nodes has a weight, which signifies the power of computation of the node.
Each node is calculated by getting the input and multiplying it by the weight and adding a bias.

```(Input * Weight) + Bias```

Calculations are implemented via matrices and matrix multiplication for the sake of efficiency.

```Error = (1/2) * ((prediction - actual) ^ 2)```

If the value of the error is equal to 0, then the prediction is accurate.

To improve the error, we can change the weights (remember that we can't change the input nor the output). This process is called 'minimization'.

- Gradient Descent: We calculate the gradient of the error function with respect to the weights and update the weights in the opposite direction of the gradient to minimize the error.
- Backpropagation: The process of calculating the gradients of the error function with respect to the weights in a neural network. It is done by applying the chain rule of calculus to calculate the gradients layer by layer, starting from the output layer and moving backward to the input layer.

# 4. Parameter-Efficient Fine-Tuning (PEFT)
### 4.1. Understanding LoRa
LoRA is a parameter-efficient fine-tuning technique that allows us to update the weights of a pre-trained model in a more efficient way by learning a low-rank approximation of the weight matrix. This can be useful when fine-tuning large models on small datasets, as it reduces the number of parameters that need to be updated.

Given a layer in a NN, freeze the original weight matrix, train a separate weight matrix, then use the new matrix to update the original matrix output:
```Wx + delta(W)```.
W and delta(W) must have the same size for the addition to work.

### 4.2. Training a Lora Model
**Implementation in PyTorch:**

- **LoRA Layer:** 
```
class Lora(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.A = torch.nn.Parameter(torch.randn(in_features, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)
```

- **Fake LLm:**
```
class FakeLLM(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = torch.nn.Linear(in_features, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, out_features)
        
    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```

- **Modified Network (Add LoRA Weights):**
```
class FakeLLMWLoRA(torch.nn.Module):
    def __init__(self, model, rank=2, alpha=0.5):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.lora_layer_1 = Lora(model.fc1.in_features, model.fc1.out_features, rank, alpha)
        self.lora_layer_2 = Lora(model.fc2.in_features, model.fc2.out_features, rank, alpha)
        self.lora_layer_3 = Lora(model.fc3.in_features, model.fc3.out_features, rank, alpha)
        
    def forward(self, x):
        x = x.view(-1, self.model.in_features)
        x = torch.relu(self.model.fc1(x) + self.lora_layer_1(x))
        x = torch.relu(self.model.fc2(x) + self.lora_layer_2(x))
        x = torch.relu(self.model.fc3(x) + self.lora_layer_3(x))
        x = self.model.fc4(x)
        return x
```

## 5. Demo Application
### Description: Fine-Tuning an LLM Using LoRA to Mask Sensitive Personally Identifiable Information (PII)

**Tech Stack:** Huggingface, Python, PyTorch, Llama Factory

### References:
- LoRA Paper: https://arxiv.org/abs/2106.09685
- Fine-Tuning Guide: https://www.lakera.ai/blog/llm-fine-tuning-guide
- Backpropagation: https://hmkcode.com/ai/backpropagation-step-by-step/

### Credits:
Ali Masri, PhD - Machine Learning Engineer
