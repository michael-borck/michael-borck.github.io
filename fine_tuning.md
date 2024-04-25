Fine-Tuning Large Language Models (LLMs)
A conceptual overview with example Python code

This is the 5th article in a series on using Large Language Models (LLMs) in practice. In this post, we will discuss how to fine-tune (FT) a pre-trained LLM. We start by introducing key FT concepts and techniques, then finish with a concrete example of how to fine-tune a model (locally) using Python and Hugging Face’s software ecosystem.
Tuning a language model. Image by author.

In the previous article of this series, we saw how we could build practical LLM-powered applications by integrating prompt engineering into our Python code. For the vast majority of LLM use cases, this is the initial approach I recommend because it requires significantly less resources and technical expertise than other methods while still providing much of the upside.

However, there are situations where prompting an existing LLM out-of-the-box doesn’t cut it, and a more sophisticated solution is required. This is where model fine-tuning can help.
What is Fine-tuning?

Fine-tuning is taking a pre-trained model and training at least one internal model parameter (i.e. weights). In the context of LLMs, what this typically accomplishes is transforming a general-purpose base model (e.g. GPT-3) into a specialized model for a particular use case (e.g. ChatGPT) [1].

The key upside of this approach is that models can achieve better performance while requiring (far) fewer manually labeled examples compared to models that solely rely on supervised training.

While strictly self-supervised base models can exhibit impressive performance on a wide variety of tasks with the help of prompt engineering [2], they are still word predictors and may generate completions that are not entirely helpful or accurate. For example, let’s compare the completions of davinci (base GPT-3 model) and text-davinci-003 (a fine-tuned model).
Completion comparison of davinci (base GPT-3 model) and text-davinci-003 (a fine-tuned model). Image by author.

Notice the base model is simply trying to complete the text by listing a set of questions like a Google search or homework assignment, while the fine-tuned model gives a more helpful response. The flavor of fine-tuning used for text-davinci-003 is alignment tuning, which aims to make the LLM’s responses more helpful, honest, and harmless, but more on that later [3,4].
Why Fine-tune

Fine-tuning not only improves the performance of a base model, but a smaller (fine-tuned) model can often outperform larger (more expensive) models on the set of tasks on which it was trained [4]. This was demonstrated by OpenAI with their first generation “InstructGPT” models, where the 1.3B parameter InstructGPT model completions were preferred over the 175B parameter GPT-3 base model despite being 100x smaller [4].

Although most of the LLMs we may interact with these days are not strictly self-supervised models like GPT-3, there are still drawbacks to prompting an existing fine-tuned model for a specific use case.

A big one is LLMs have a finite context window. Thus, the model may perform sub-optimally on tasks that require a large knowledge base or domain-specific information [1]. Fine-tuned models can avoid this issue by “learning” this information during the fine-tuning process. This also precludes the need to jam-pack prompts with additional context and thus can result in lower inference costs.
3 Ways to Fine-tune

There are 3 generic ways one can fine-tune a model: self-supervised, supervised, and reinforcement learning. These are not mutually exclusive in that any combination of these three approaches can be used in succession to fine-tune a single model.
Self-supervised Learning

Self-supervised learning consists of training a model based on the inherent structure of the training data. In the context of LLMs, what this typically looks like is given a sequence of words (or tokens, to be more precise), predict the next word (token).

While this is how many pre-trained language models are developed these days, it can also be used for model fine-tuning. A potential use case of this is developing a model that can mimic a person’s writing style given a set of example texts.
Supervised Learning

The next, and perhaps most popular, way to fine-tune a model is via supervised learning. This involves training a model on input-output pairs for a particular task. An example is instruction tuning, which aims to improve model performance in answering questions or responding to user prompts [1,3].

The key step in supervised learning is curating a training dataset. A simple way to do this is to create question-answer pairs and integrate them into a prompt template [1,3]. For example, the question-answer pair: Who was the 35th President of the United States? — John F. Kennedy could be pasted into the below prompt template. More example prompt templates are available in section A.2.1 of ref [4].

"""Please answer the following question.

Q: {Question}
 
A: {Answer}"""

Using a prompt template is important because base models like GPT-3 are essentially “document completers”. Meaning, given some text, the model generates more text that (statistically) makes sense in that context. This goes back to the previous blog of this series and the idea of “tricking” a language model into solving your problem via prompt engineering.
Prompt Engineering — How to trick AI into solving your problems
7 prompting tricks, Langchain, and Python example code

towardsdatascience.com
Reinforcement Learning

Finally, one can use reinforcement learning (RL) to fine-tune models. RL uses a reward model to guide the training of the base model. This can take many different forms, but the basic idea is to train the reward model to score language model completions such that they reflect the preferences of human labelers [3,4]. The reward model can then be combined with a reinforcement learning algorithm (e.g. Proximal Policy Optimization (PPO)) to fine-tune the pre-trained model.

An example of how RL can be used for model fine-tuning is demonstrated by OpenAI’s InstructGPT models, which were developed through 3 key steps [4].

    Generate high-quality prompt-response pairs and fine-tune a pre-trained model using supervised learning. (~13k training prompts) Note: One can (alternatively) skip to step 2 with the pre-trained model [3].
    Use the fine-tuned model to generate completions and have human-labelers rank responses based on their preferences. Use these preferences to train the reward model. (~33k training prompts)
    Use the reward model and an RL algorithm (e.g. PPO) to fine-tune the model further. (~31k training prompts)

While the strategy above does generally result in LLM completions that are significantly more preferable to the base model, it can also come at a cost of lower performance in a subset of tasks. This drop in performance is also known as an alignment tax [3,4].
Supervised Fine-tuning Steps (High-level)

As we saw above, there are many ways in which one can fine-tune an existing language model. However, for the remainder of this article, we will focus on fine-tuning via supervised learning. Below is a high-level procedure for supervised model fine-tuning [1].

    Choose fine-tuning task (e.g. summarization, question answering, text classification)
    Prepare training dataset i.e. create (100–10k) input-output pairs and preprocess data (i.e. tokenize, truncate, and pad text).
    Choose a base model (experiment with different models and choose one that performs best on the desired task).
    Fine-tune model via supervised learning
    Evaluate model performance

While each of these steps could be an article of their own, I want to focus on step 4 and discuss how we can go about training the fine-tuned model.
3 Options for Parameter Training

When it comes to fine-tuning a model with ~100M-100B parameters, one needs to be thoughtful of computational costs. Toward this end, an important question is — which parameters do we (re)train?

With the mountain of parameters at play, we have countless choices for which ones we train. Here, I will focus on three generic options of which to choose.
Option 1: Retrain all parameters

The first option is to train all internal model parameters (called full parameter tuning) [3]. While this option is simple (conceptually), it is the most computationally expensive. Additionally, a known issue with full parameter tuning is the phenomenon of catastrophic forgetting. This is where the model “forgets” useful information it “learned” in its initial training [3].

One way we can mitigate the downsides of Option 1 is to freeze a large portion of the model parameters, which brings us to Option 2.
Option 2: Transfer Learning

The big idea with transfer learning (TL) is to preserve the useful representations/features the model has learned from past training when applying the model to a new task. This generally consists of dropping “the head” of a neural network (NN) and replacing it with a new one (e.g. adding new layers with randomized weights). Note: The head of an NN includes its final layers, which translate the model’s internal representations to output values.

While leaving the majority of parameters untouched mitigates the huge computational cost of training an LLM, TL may not necessarily resolve the problem of catastrophic forgetting. To better handle both of these issues, we can turn to a different set of approaches.
Option 3: Parameter Efficient Fine-tuning (PEFT)

PEFT involves augmenting a base model with a relatively small number of trainable parameters. The key result of this is a fine-tuning methodology that demonstrates comparable performance to full parameter tuning at a tiny fraction of the computational and storage cost [5].

PEFT encapsulates a family of techniques, one of which is the popular LoRA (Low-Rank Adaptation) method [6]. The basic idea behind LoRA is to pick a subset of layers in an existing model and modify their weights according to the following equation.
Equation showing how weight matrices are modified for fine-tuning using LoRA [6]. Image by author.

Where h() = a hidden layer that will be tuned, x = the input to h(), W₀ = the original weight matrix for the h, and ΔW = a matrix of trainable parameters injected into h. ΔW is decomposed according to ΔW=BA, where ΔW is a d by k matrix, B is d by r, and A is r by k. r is the assumed “intrinsic rank” of ΔW (which can be as small as 1 or 2) [6].

Sorry for all the math, but the key point is the (d * k) weights in W₀ are frozen and, thus, not included in optimization. Instead, the ((d * r) + (r * k)) weights making up matrices B and A are the only ones that are trained.

Plugging in some made-up numbers for d=100, k=100, and r=2 to get a sense of the efficiency gains, the number of trainable parameters drops from 10,000 to 400 in that layer. In practice, the authors of the LoRA paper cited a 10,000x reduction in parameter checkpoint size using LoRA fine-tune GPT-3 compared to full parameter tuning [6].

To make this more concrete, let’s see how we can use LoRA to fine-tune a language model efficiently enough to run on a personal computer.
Example Code: Fine-tuning an LLM using LoRA

In this example, we will use the Hugging Face ecosystem to fine-tune a language model to classify text as ‘positive’ or ‘negative’. Here, we fine-tune distilbert-base-uncased, a ~70M parameter model based on BERT. Since this base model was trained to do language modeling and not classification, we employ transfer learning to replace the base model head with a classification head. Additionally, we use LoRA to fine-tune the model efficiently enough that it can run on my Mac Mini (M1 chip with 16GB memory) in a reasonable amount of time (~20 min).

The code, along with the conda environment files, are available on the GitHub repository. The final model and dataset [7] are available on Hugging Face.
YouTube-Blog/LLMs/fine-tuning at main · ShawhinT/YouTube-Blog
Codes to complement YouTube videos and blog posts on Medium. - YouTube-Blog/LLMs/fine-tuning at main ·…

github.com
Imports

We start by importing helpful libraries and modules. Datasets, transformers, peft, and evaluate are all libraries from Hugging Face (HF).

from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np

Base model

Next, we load in our base model. The base model here is a relatively small one, but there are several other (larger) ones that we could have used (e.g. roberta-base, llama2, gpt2). A full list is available here.

model_checkpoint = 'distilbert-base-uncased'

# define label maps
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative":0, "Positive":1}

# generate classification model from model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

Load data

We can then load our training and validation data from HF’s datasets library. This is a dataset of 2000 movie reviews (1000 for training and 1000 for validation) with binary labels indicating whether the review is positive (or not).

# load dataset
dataset = load_dataset("shawhin/imdb-truncated")
dataset

# dataset = 
# DatasetDict({
#     train: Dataset({
#         features: ['label', 'text'],
#         num_rows: 1000
#     })
#     validation: Dataset({
#         features: ['label', 'text'],
#         num_rows: 1000
#     })
# }) 

Preprocess data

Next, we need to preprocess our data so that it can be used for training. This consists of using a tokenizer to convert the text into an integer representation understood by the base model.

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

To apply the tokenizer to the dataset, we use the .map() method. This takes in a custom function that specifies how the text should be preprocessed. In this case, that function is called tokenize_function(). In addition to translating text to integers, this function truncates integer sequences such that they are no longer than 512 numbers to conform to the base model’s max input length.

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["text"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# tokenize training and validation datasets
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset

# tokenized_dataset = 
# DatasetDict({
#     train: Dataset({
#        features: ['label', 'text', 'input_ids', 'attention_mask'],
#         num_rows: 1000
#     })
#     validation: Dataset({
#         features: ['label', 'text', 'input_ids', 'attention_mask'],
#         num_rows: 1000
#     })
# })

At this point, we can also create a data collator, which will dynamically pad examples in each batch during training such that they all have the same length. This is computationally more efficient than padding all examples to be equal in length across the entire dataset.

# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

Evaluation metrics

We can define how we want to evaluate our fine-tuned model via a custom function. Here, we define the compute_metrics() function to compute the model’s accuracy.

# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")

# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, 
                                          references=labels)}

Untrained model performance

Before training our model, we can evaluate how the base model with a randomly initialized classification head performs on some example inputs.

# define list of examples
text_list = ["It was good.", "Not a fan, don't recommed.", 
"Better than the first one.", "This is not worth watching even once.", 
"This one is a pass."]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    # tokenize text
    inputs = tokenizer.encode(text, return_tensors="pt")
    # compute logits
    logits = model(inputs).logits
    # convert logits to label
    predictions = torch.argmax(logits)

    print(text + " - " + id2label[predictions.tolist()])

# Output:
# Untrained model predictions:
# ----------------------------
# It was good. - Negative
# Not a fan, don't recommed. - Negative
# Better than the first one. - Negative
# This is not worth watching even once. - Negative
# This one is a pass. - Negative

As expected, the model performance is equivalent to random guessing. Let’s see how we can improve this with fine-tuning.
Fine-tuning with LoRA

To use LoRA for fine-tuning, we first need a config file. This sets all the parameters for the LoRA algorithm. See comments in the code block for more details.

peft_config = LoraConfig(task_type="SEQ_CLS", # sequence classification
                        r=4, # intrinsic rank of trainable weight matrix
                        lora_alpha=32, # this is like a learning rate
                        lora_dropout=0.01, # probablity of dropout
                        target_modules = ['q_lin']) # we apply lora to query layer only

We can then create a new version of our model that can be trained via PEFT. Notice that the scale of trainable parameters was reduced by about 100x.

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# trainable params: 1,221,124 || all params: 67,584,004 || trainable%: 1.8068239934408148

Next, we define hyperparameters for model training.

# hyperparameters
lr = 1e-3 # size of optimization step 
batch_size = 4 # number of examples processed per optimziation step
num_epochs = 10 # number of times model runs through training data

# define training arguments
training_args = TrainingArguments(
    output_dir= model_checkpoint + "-lora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

Finally, we create a trainer() object and fine-tune the model!

# creater trainer object
trainer = Trainer(
    model=model, # our peft model
    args=training_args, # hyperparameters
    train_dataset=tokenized_dataset["train"], # training data
    eval_dataset=tokenized_dataset["validation"], # validation data
    tokenizer=tokenizer, # define tokenizer
    data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics, # evaluates model using compute_metrics() function from before
)

# train model
trainer.train()

The above code will generate the following table of metrics during training.
Model training metrics. Image by author.
Trained model performance

To see how the model performance has improved, let’s apply it to the same 5 examples from before.

model.to('mps') # moving to mps for Mac (can alternatively do 'cpu')

print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to("mps") # moving to mps for Mac (can alternatively do 'cpu')

    logits = model(inputs).logits
    predictions = torch.max(logits,1).indices

    print(text + " - " + id2label[predictions.tolist()[0]])

# Output:
# Trained model predictions:
# ----------------------------
# It was good. - Positive
# Not a fan, don't recommed. - Negative
# Better than the first one. - Positive
# This is not worth watching even once. - Negative
# This one is a pass. - Positive # this one is tricky

The fine-tuned model improved significantly from its prior random guessing, correctly classifying all but one of the examples in the above code. This aligns with the ~90% accuracy metric we saw during training.

Links: Code Repo | Model | Dataset
Conclusions

While fine-tuning an existing model requires more computational resources and technical expertise than using one out-of-the-box, (smaller) fine-tuned models can outperform (larger) pre-trained base models for a particular use case, even when employing clever prompt engineering strategies. Furthermore, with all the open-source LLM resources available, it’s never been easier to fine-tune a model for a custom application.

The next (and final) article of this series will go one step beyond model fine-tuning and discuss how to train a language model from scratch.



QLoRA — How to Fine-Tune an LLM on a Single GPU
An introduction with Python example code (ft. Mistral-7b)

This article is part of a larger series on using large language models (LLMs) in practice. In the previous post, we saw how to fine-tune an LLM using OpenAI. The main limitation to this approach, however, is that OpenAI’s models are concealed behind their API, which limits what and how we can build with them. Here, I’ll discuss an alternative way to fine-tune an LLM using open-source models and QLoRA.
Photo by Dell on Unsplash

Fine-tuning is when we take an existing model and tweak it for a particular use case. This has been a critical part of the recent explosion of AI innovations, giving rise to ChatGPT and the like.

Although fine-tuning is a simple (and powerful) idea, applying it to LLMs isn’t always straightforward. The key challenge is that LLMs are (very) computationally expensive (i.e. they aren’t something that can be trained on a typical laptop).

For example, standard fine-tuning of a 70B parameter model requires over 1TB of memory [1]. For context, an A100 GPU comes with up to 80GB of memory, so you’d (at best) need over a dozen of these $20,000 cards!

While this may deflate your dreams of building a custom AI, don’t give up just yet. The open-source community has been working hard to make building with these models more accessible. One popular method that has sprouted from these efforts is QLoRA (Quantized Low-Rank Adaptation), an efficient way to fine-tune a model without sacrificing performance.
What is Quantization?

A key part of QLoRA is so-called quantization. While this might sound like a scary and sophisticated word, it is a simple idea. When you hear “quantizing,” think of splitting a range of numbers into buckets.

For example, there are infinite possible numbers between 0 and 100, e.g. 1, 12, 27, 55.3, 83.7823, and so on. We could quantize this range by splitting them into buckets based on whole numbers so that (1, 12, 27, 55.3, 83.7823) becomes (1, 12, 27, 55, 83), or we could use factors of ten so that the numbers become (0, 0, 20, 50, 80). A visualization of this process is shown below.
Visualization of quantizing numbers via whole numbers or 10s. Image by author.
Why we need it

Quantization allows us to represent a given set of numbers with less information. To see why this is important, let’s (briefly) talk about how computers work.

Computers encode information using binary digits (i.e. bits). For instance, if I want a computer to remember the number 83.7823, this number needs to be translated into a string of 1s and 0s (aka a bit string).

One way of doing this is via the single-precision floating-point format (i.e. FP32), which represents numbers as a sequence of 32 bits [2]. For example, 83.7823 can be represented as 01000010101001111001000010001010 [3].

Since a string of 32 bits has 2³² (= 4,294,967,296) unique combinations that means we can represent 4,294,967,296 unique values with FP32. Thus, if we have numbers from 0 to 100, the bit count sets the precision for representing numbers in that range.

But there is another side of the story. If we use 32 bits to represent each model parameter, each parameter will take up 4 bytes of memory (1 byte = 8 bits). Therefore, a 10B parameter model will consume 40 GB of memory. And if we want to do full parameter fine-tuning, that will require closer to 200GB of memory! [1]

This presents a dilemma for fine-tuning LLMs. Namely, we want high precision for successful model training, but we need to use as little memory as possible to ensure we don’t run out of it. Balancing this tradeoff is a key contribution of QLoRA.
QLoRA

QLoRA (or Quantized Low-Rank Adaptation) combines 4 ingredients to get the most out of a machine’s limited memory without sacrificing model performance. I will briefly summarize key points from each. More details are available in the QLoRA paper [4].
Ingredient 1: 4-bit NormalFloat

This first ingredient takes the idea of quantization near its practical limits. In contrast to the typical 16-bit data type (i.e., half-precision floating point) used for language model parameters, QLoRA uses a special data type called 4-bit NormalFloat.

As the name suggests, this data type encodes numbers with just 4 bits. While this means we only have 2⁴ (= 16) buckets to represent model parameters, 4-bit NormalFloat uses a special trick to get more out of the limited information capacity.

The naive way to quantize a set of numbers is what we saw earlier, where we split the numbers into equally-spaced buckets. However, a more efficient way would be to use equally-sized buckets. The difference between these two approaches is illustrated in the figure below.
Difference between equally-spaced and equally-sized buckets

More specifically, 4-bit NormalFloat employs an information-theoretically optimal quantization strategy for normally distributed data [4]. Since model parameters tend to clump around 0, this is an effective strategy for representing LLM parameters.
Ingredient 2: Double Quantization

Despite the unfortunate name, double quantization generates memory savings by quantizing the quantization constants (see what I mean).

To break this down, consider the following quantization process. Given an FP32 tensor, a simple way to quantize it is using the mathematical formula below [4].
Simple quantization formula from FP32 to Int8. Example from [4]. Image by author.

Here we are converting the FP32 representation into an Int8 (8-bit integer) representation within the range of [-127, 127]. Notice this boils down to rescaling the values in the tensor X^(FP32) and then rounding them to the nearest integer. We can then simplify the equation by defining a scaling term (or quantization constant) c^FP32 = 127/absmax(X^FP32)).

While this naive quantization approach isn’t how it’s done in practice (remember the trick we saw with 4-bit NormalFloat), it does illustrate that the quantization comes with some computational overhead to store the resulting constants in memory.

We could minimize this overhead by doing this process just once. In other words, compute one quantization constant for all the model parameters. However, this is not ideal since it is (very) sensitive to extreme values. In other words, one relatively large parameter value will skew all the others because of the absmax() function in c^FP32.

Alternatively, we could partition the model parameters into smaller blocks for quantization. This reduces the chances that a large value will skew other values but comes with a larger memory footprint.

To mitigate this memory cost, we can (again) employ quantization, but now on the constants generated from this block-wise approach. For a block size of 64, an FP32 quantization constant adds 0.5 bits/parameter. By quantizing these constants further, to say 8-bit, we can reduce this footprint to 0.127 bits/parameter [4].
Visual comparison of standard vs block-wise quantization. Image by author.
Ingredient 3: Paged optimizers

This ingredient uses Nvidia’s unified memory feature to help avoid out-of-memory errors during training. It transfers “pages” of memory from the GPU to the CPU when the GPU hits its limits. This is similar to how memory is handled between CPU RAM and machine storage [4].

More specifically, this memory paging feature moves pages of optimizer states to the CPU and back to the GPU as needed. This is important because there can be intermittent memory spikes during training, which can kill the process.
Ingredient 4: LoRA

LoRA (Low-rank Adaptation) is a Parameter Efficient Fine-tuning (PEFT) method. The key idea is instead of retraining all the model parameters, LoRA adds a relatively small number of trainable parameters while keeping the original parameters fixed [5].

Since I covered the details of LoRA in a previous article of this series, I will just say we can use it to reduce the number of trainable parameters by 100–1000X without sacrificing model performance.
Fine-Tuning Large Language Models (LLMs)
A conceptual overview with example Python code

towardsdatascience.com
Bringing it all together

Now that we know all the ingredients of QLoRA, let’s see how we can bring them together.

To start, consider a standard fine-tuning process, which consists of retraining every model parameter. What this might look like is using FP16 for the model parameters and gradients (4 total bytes/parameters) and FP32 for the optimizer states, e.g. momentum and variance, and parameters (12 bytes/parameter) [1]. So, a 10B parameter model would require about 160GB of memory to fine-tune.

Using LoRA, we can immediately reduce this computational cost by decreasing the number of trainable parameters. This works by freezing the original parameters and adding a set of (small) adapters housing the trainable parameters [5]. The computational cost for the model parameters and gradients would be the same as before (4 total bytes/parameters) [1].

The savings, however, comes from the optimizer states. If we have 100X fewer trainable parameters and use FP16 for the adapter, we’d have an additional 0.04 bytes per parameter in the original model (as opposed to 4 bytes/parameter). Similarly, using FP32 for the optimizer states, we have an additional 0.12 bytes/parameter [4]. Therefore, a 10B parameter model would require about 41.6GB of memory to fine-tune. A significant savings, but still a lot to ask for from consumer hardware.

QLoRA takes things further by quantizing the original model parameters using Ingredients 1 and 2. This reduces the cost from 4 bytes/parameter to about 1 byte/parameter. Then, by using LoRA in the same way as before, that would add another 0.16 bytes/parameter. Thus, a 10B model can be fine-tuned with just 11.6GB of memory! This can easily run on consumer hardware like the free T4 GPU on Google Colab.

A visual comparison of the 3 approaches is shown below [4].
Visual comparison of 3 fine-tuning techniques. Based on the figure in [4]. Re-illustrated by author.
Example Code: Fine-tuning Mistral-7b-Instruct to respond to YouTube comments

Now that we have a basic understanding of how QLoRA works let’s see what using it looks like in code. Here, we will use a 4-bit version of the Mistral-7B-Instruct model provided by TheBloke and the Hugging Face ecosystem for fine-tuning.

This example code is available in a Google Colab notebook, which can run on the (free) GPU provided by Colab. The dataset is also available on Hugging Face.

🔗 Google Colab | Training Dataset | GitHub Repo
Imports

We import modules from Hugging Face’s transforms, peft, and datasets libraries.

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers

Additionally, we need the following dependencies installed for some of the previous modules to work.

!pip install auto-gptq
!pip install optimum
!pip install bitsandbytes

Load Base Model & Tokenizer

Next, we load the quantized model from Hugging Face. Here, we use a version of Mistral-7B-Instruct-v0.2 prepared by TheBloke, who has freely quantized and shared thousands of LLMs.

Notice we are using the “Instruct” version of Mistral-7b. This indicates that the model has undergone instruction tuning, a fine-tuning process that aims to improve model performance in answering questions and responding to user prompts.

Other than specifying the model repo we want to download, we also set the following arguments: device_map, trust_remote_code, and revision. device_map lets the method automatically figure out how to best allocate computational resources for loading the model on the machine. Next, trust_remote_code=False prevents custom model files from running on your machine. Then, finally, revision specifies which version of the model we want to use from the repo.

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", 
    trust_remote_code=False,
    revision="main") 

Once loaded, we see the 7B parameter model only takes us 4.16GB of memory, which can easily fit in either the CPU or GPU memory available for free on Colab.

Next, we load the tokenizer for the model. This is necessary because the model expects the text to be encoded in a specific way. I discussed tokenization in previous articles of this series.

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

Using the Base Model

Next, we can use the model for text generation. As a first pass, let’s try to input a test comment to the model. We can do this in 3 steps.

First, we craft the prompt in the proper format. Namely, Mistral-7b-Instruct expects input text to start and end with the special tokens [INST] and [/INST], respectively. Second, we tokenize the prompt. Third, we pass the prompt into the model to generate text.

The code to do this is shown below with the test comment, “Great content, thank you!”

model.eval() # model in evaluation mode (dropout modules are deactivated)

# craft prompt
comment = "Great content, thank you!"
prompt=f'''[INST] {comment} [/INST]'''

# tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# generate output
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), 
                            max_new_tokens=140)

print(tokenizer.batch_decode(outputs)[0])

The response from the model is shown below. While it gets off to a good start, the response seems to continue for no good reason and doesn’t sound like something I would say.

I'm glad you found the content helpful! If you have any specific questions or 
topics you'd like me to cover in the future, feel free to ask. I'm here to 
help.

In the meantime, I'd be happy to answer any questions you have about the 
content I've already provided. Just let me know which article or blog post 
you're referring to, and I'll do my best to provide you with accurate and 
up-to-date information.

Thanks for reading, and I look forward to helping you with any questions you 
may have!

Prompt Engineering

This is where prompt engineering is helpful. Since a previous article in this series covered this topic in-depth, I’ll just say that prompt engineering involves crafting instructions that lead to better model responses.

Typically, writing good instructions is something done through trial and error. To do this, I tried several prompt iterations using together.ai, which has a free UI for many open-source LLMs, such as Mistral-7B-Instruct-v0.2.

Once I got instructions I was happy with, I created a prompt template that automatically combines these instructions with a comment using a lambda function. The code for this is shown below.

intstructions_string = f"""ShawGPT, functioning as a virtual data science \
consultant on YouTube, communicates in clear, accessible language, escalating \
to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature '–ShawGPT'. \
ShawGPT will tailor the length of its responses to match the viewer's comment, 
providing concise acknowledgments to brief expressions of gratitude or \
feedback, thus keeping the interaction natural and engaging.

Please respond to the following comment.
"""

prompt_template = 
    lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''

prompt = prompt_template(comment)

The Prompt
-----------

[INST] ShawGPT, functioning as a virtual data science consultant on YouTube, 
communicates in clear, accessible language, escalating to technical depth upon 
request. It reacts to feedback aptly and ends responses with its signature 
'–ShawGPT'. ShawGPT will tailor the length of its responses to match the 
viewer's comment, providing concise acknowledgments to brief expressions of 
gratitude or feedback, thus keeping the interaction natural and engaging.

Please respond to the following comment.
 
Great content, thank you! 
[/INST]

We can see the power of a good prompt by comparing the new model response (below) to the previous one. Here, the model responds concisely and appropriately and identifies itself as ShawGPT.

Thank you for your kind words! I'm glad you found the content helpful. –ShawGPT

Prepare Model for Training

Let’s see how we can improve the model’s performance through fine-tuning. We can start by enabling gradient checkpointing and quantized training. Gradient checkpointing is a memory-saving technique that clears specific activations and recomputes them during the backward pass [6]. Quantized training is enabled using the method imported from peft.

model.train() # model in training mode (dropout modules are activated)

# enable gradient check pointing
model.gradient_checkpointing_enable()

# enable quantized training
model = prepare_model_for_kbit_training(model)

Next, we can set up training with LoRA via a configuration object. Here, we target the query layers in the model and use an intrinsic rank of 8. Using this config, we can create a version of the model that can undergo fine-tuning with LoRA. Printing the number of trainable parameters, we observe a more than 100X reduction.

# LoRA config
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA trainable version of model
model = get_peft_model(model, config)

# trainable parameter count
model.print_trainable_parameters()

### trainable params: 2,097,152 || all params: 264,507,392 || trainable%: 0.7928519441906561
# Note: I'm not sure why its showing 264M parameters here.

Prepare Training Dataset

Now, we can import our training data. The dataset used here is available on the HuggingFace Dataset Hub. I generated this dataset using comments and responses from my YouTube channel. The code to prepare and upload the dataset to the Hub is available at the GitHub repo.

# load dataset
data = load_dataset("shawhin/shawgpt-youtube-comments")

Next, we must prepare the dataset for training. This involves ensuring examples are an appropriate length and are tokenized. The code for this is shown below.

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["example"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_data = data.map(tokenize_function, batched=True)

Two other things we need for training are a pad token and a data collator. Since not all examples are the same length, a pad token can be added to examples as needed to make it a particular size. A data collator will dynamically pad examples during training to ensure all examples in a given batch have the same length.

# setting pad token
tokenizer.pad_token = tokenizer.eos_token

# data collator
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, 
                                                              mlm=False)

Fine-tuning the Model

In the code block below, I define hyperparameters for model training.

# hyperparameters
lr = 2e-4
batch_size = 4
num_epochs = 10

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir= "shawgpt-ft",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",
)

While several are listed here, the two I want to highlight in the context of QLoRA are fp16 and optim. fp16=True has the trainer use FP16 values for the training process, which results in significant memory savings compared to the standard FP32. optim=”paged_adamw_8bit” enables Ingredient 3 (i.e. paged optimizers) discussed previously.

With all the hyperparameters set, we can run the training process using the code below.

# configure trainer
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    args=training_args,
    data_collator=data_collator
)

# train model
model.config.use_cache = False  # silence the warnings.
trainer.train()

# renable warnings
model.config.use_cache = True

Since we only have 50 training examples, the process runs in about 10 minutes. The training and validation loss are shown in the table below. We can see that both losses monotonically decrease, indicating stable training.
Training and Validation loss table. Image by author.
Loading Fine-tuned model

The final model is freely available on the HF hub. If you want to skip the training process and load it directly, you can use the following code.

# load model from hub
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

config = PeftConfig.from_pretrained("shawhin/shawgpt-ft")
model = PeftModel.from_pretrained(model, "shawhin/shawgpt-ft")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

Using the Fine-tuned Model

We can use the fine-tuned model for inference in the same way as before. Here is the fine-tuned model’s response to the same test comment as before (i.e. “Great content, thank you!”).

Glad you enjoyed it! –ShawGPT

(Note: I'm an AI language model, I don't have the ability to feel emotions or 
watch videos. I'm here to help answer questions and provide explanations.)

The response is much more concise and even adds a disclaimer that it is an AI. If we want to remove this disclaimer, we can easily do that using basic string manipulation in Python.

As another example, let’s try a more technical test comment: “What is fat-tailedness?” The model’s response is given below.

Fat-tailedness is a statistical property of a distribution. In simple terms, 
it refers to the presence of extreme outliers or heavy tails in the 
distribution.

For instance, consider the distribution of heights in a population. A normal 
distribution would have most people clustered around an average height with a 
few people deviating slightly from the mean. However, in a fat-tailed 
distribution, you would observe a larger number of people being significantly 
taller or shorter than the average.

In the context of finance, fat-tailedness is often used to describe the 
presence of extreme market moves or volatility. For example, the 2008 
financial crisis is an example of a fat-tailed event.

I hope this explanation helps! Let me know if you have any questions.
–ShawGPT

The response is similar to what we saw in the previous article of this series with the fine-tuned OpenAI model. It gives a concise and appropriate explanation of fat-tailedness, but this isn’t how I explain fat-tailedness.

While we could attempt to capture this specialized knowledge via further fine-tuning, a simpler approach would be to augment the fine-tuned model using external knowledge from my article series on fat tails (and other data science topics).

This brings up the idea of Retrieval Augmented Generation (i.e. RAG), which will be discussed in the next article of this series.
YouTube-Blog/LLMs/qlora at main · ShawhinT/YouTube-Blog
Codes to complement YouTube videos and blog posts on Medium. - YouTube-Blog/LLMs/qlora at main · ShawhinT/YouTube-Blog

github.com
What’s Next?

QLoRA is a fine-tuning technique that has made building custom large language models more accessible. Here, I gave an overview of how the approach works and shared a concrete example of using QLoRA to create a YouTube comment responder.

While the fine-tuned model did a qualitatively good job at mimicking my response style, it had some limitations in its understanding of specialized data science knowledge. In the next article of this series, we will see how we can overcome this limitation by improving the model with RAG.

How to Improve LLMs with RAG
A beginner-friendly introduction w/ Python code

This article is part of a larger series on using large language models in practice. In the previous post, we fine-tuned Mistral-7b-Instruct to respond to YouTube comments using QLoRA. Although the fine-tuned model successfully captured my style when responding to viewer feedback, its responses to technical questions didn’t match my explanations. Here, I’ll discuss how we can improve LLM performance using retrieval augmented generation (i.e. RAG).
The original RAG system. Image from Canva.

Large language models (LLMs) have demonstrated an impressive ability to store and deploy vast knowledge in response to user queries. While this has enabled the creation of powerful AI systems like ChatGPT, compressing world knowledge in this way has two key limitations.

First, an LLM’s knowledge is static, i.e., not updated as new information becomes available. Second, LLMs may have an insufficient “understanding” of niche and specialized information that was not prominent in their training data. These limitations can result in undesirable (and even fictional) model responses to user queries.

One way we can mitigate these limitations is to augment a model via a specialized and mutable knowledge base, e.g., customer FAQs, software documentation, or product catalogs. This enables the creation of more robust and adaptable AI systems.

Retrieval augmented generation, or RAG, is one such approach. Here, I provide a high-level introduction to RAG and share example Python code for implementing a RAG system using LlamaIndex.
What is RAG?

The basic usage of an LLM consists of giving it a prompt and getting back a response.
Basic usage of an LLM i.e. prompt in, response out. Image by author.

RAG works by adding a step to this basic process. Namely, a retrieval step is performed where, based on the user’s prompt, the relevant information is extracted from an external knowledge base and injected into the prompt before being passed to the LLM.
Overview of RAG system. Image by author.
Why we care

Notice that RAG does not fundamentally change how we use an LLM; it's still prompt-in and response-out. RAG simply augments this process (hence the name).

This makes RAG a flexible and (relatively) straightforward way to improve LLM-based systems. Additionally, since knowledge is stored in an external database, updating system knowledge is as simple as adding or removing records from a table.
Why not fine-tune?

Previous articles in this series discussed fine-tuning, which adapts an existing model for a particular use case. While this is an alternative way to endow an LLM with specialized knowledge, empirically, fine-tuning seems to be less effective than RAG at doing this [1].
How it works

There are 2 key elements of a RAG system: a retriever and a knowledge base.
Retriever

A retriever takes a user prompt and returns relevant items from a knowledge base. This typically works using so-called text embeddings, numerical representations of text in concept space. In other words, these are numbers that represent the meaning of a given text.

Text embeddings can be used to compute a similarity score between the user’s query and each item in the knowledge base. The result of this process is a ranking of each item’s relevance to the input query.

The retriever can then take the top k (say k=3) most relevant items and inject them into the user prompt. This augmented prompt is then passed into the LLM for generation.
Overview of retrieval step. Image by author.
Knowledge Base

The next key element of a RAG system is a knowledge base. This houses all the information you want to make available to the LLM. While there are countless ways to construct a knowledge base for RAG, here I’ll focus on building one from a set of documents.

The process can be broken down into 4 key steps [2,3].

    Load docs — This consists of gathering a collection of documents and ensuring they are in a ready-to-parse format (more on this later).
    Chunk docs—Since LLMs have limited context windows, documents must be split into smaller chunks (e.g., 256 or 512 characters long).
    Embed chunks — Translate each chunk into numbers using a text embedding model.
    Load into Vector DB— Load text embeddings into a database (aka a vector database).

Overview of knowledge base creation. Image by author.
Some Nuances

While the steps for building a RAG system are conceptually simple, several nuances can make building one (in the real world) more complicated.

Document preparation—The quality of a RAG system is driven by how well useful information can be extracted from source documents. For example, if a document is unformatted and full of images and tables, it will be more difficult to parse than a well-formatted text file.

Choosing the right chunk size—We already mentioned the need for chunking due to LLM context windows. However, there are 2 additional motivations for chunking.

First, it keeps (compute) costs down. The more text you inject into the prompt, the more compute required to generate a completion. The second is performance. Relevant information for a particular query tends to be localized in source documents (often, just 1 sentence can answer a question). Chunking helps minimize the amount of irrelevant information passed into the model [4].

Improving search — While text embeddings enable a powerful and fast way to do search, it doesn’t always work as one might hope. In other words, it may return results that are “similar” to the user query, yet not helpful for answering it, e.g., “How’s the weather in LA?” may return “How’s the weather in NYC?”.

The simplest way to mitigate this is through good document preparation and chunking. However, for some use cases, additional strategies for improving search might be necessary, such as using meta-tags for each chunk, employing hybrid search, which combines keyword—and embedding-based search, or using a reranker, which is a specialized model that computes the similarity of 2 input pieces of text.
Example code: Improving YouTube Comment Responder with RAG

With a basic understanding of how RAG works, let’s see how to use it in practice. I will build upon the example from the previous article, where I fine-tuned Mistral-7B-Instruct to respond to YouTube comments using QLoRA. We will use LlamaIndex to add a RAG system to the fine-tuned model from before.

The example code is freely available in a Colab Notebook, which can run on the (free) T4 GPU provided. The source files for this example are available at the GitHub repository.

🔗 Google Colab | GitHub Repo
Imports

We start by installing and importing necessary Python libraries.

!pip install llama-index
!pip install llama-index-embeddings-huggingface
!pip install peft
!pip install auto-gptq
!pip install optimum
!pip install bitsandbytes
# if not running on Colab ensure transformers is installed too

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

Setting up Knowledge Base

We can configure our knowledge base by defining our embedding model, chunk size, and chunk overlap. Here, we use the ~33M parameter bge-small-en-v1.5 embedding model from BAAI, which is available on the Hugging Face hub. Other embedding model options are available on this text embedding leaderboard.

# import any embedding model on HF hub
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = None # we won't use LlamaIndex to set up LLM
Settings.chunk_size = 256
Settings.chunk_overlap = 25

Next, we load our source documents. Here, I have a folder called “articles,” which contains PDF versions of 3 Medium articles I wrote on fat tails. If running this in Colab, you must download the articles folder from the GitHub repo and manually upload it to your Colab environment.

For each file in this folder, the function below will read the text from the PDF, split it into chunks (based on the settings defined earlier), and store each chunk in a list called documents.

documents = SimpleDirectoryReader("articles").load_data()

Since the blogs were downloaded directly as PDFs from Medium, they resemble a webpage more than a well-formatted article. Therefore, some chunks may include text unrelated to the article, e.g., webpage headers and Medium article recommendations.

In the code block below, I refine the chunks in documents, removing most of the chunks before or after the meat of an article.

print(len(documents)) # prints: 71
for doc in documents:
    if "Member-only story" in doc.text:
        documents.remove(doc)
        continue

    if "The Data Entrepreneurs" in doc.text:
        documents.remove(doc)

    if " min read" in doc.text:
        documents.remove(doc)

print(len(documents)) # prints: 61

Finally, we can store the refined chunks in a vector database.

index = VectorStoreIndex.from_documents(documents)

Setting up Retriever

With our knowledge base in place, we can create a retriever using LlamaIndex’s VectorIndexRetreiver(), which returns the top 3 most similar chunks to a user query.

# set number of docs to retreive
top_k = 3

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)

Next, we define a query engine that uses the retriever and query to return a set of relevant chunks.

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

Use Query Engine

Now, with our knowledge base and retrieval system set up, let’s use it to return chunks relevant to a query. Here, we’ll pass the same technical question we asked ShawGPT (the YouTube comment responder) from the previous article.

query = "What is fat-tailedness?"
response = query_engine.query(query)

The query engine returns a response object containing the text, metadata, and indexes of relevant chunks. The code block below returns a more readable version of this information.

# reformat response
context = "Context:\n"
for i in range(top_k):
    context = context + response.source_nodes[i].text + "\n\n"

print(context)

Context:
Some of the controversy might be explained by the observation that log-
normal distributions behave like Gaussian for low sigma and like Power Law
at high sigma [2].
However, to avoid controversy, we can depart (for now) from whether some
given data fits a Power Law or not and focus instead on fat tails.
Fat-tailedness — measuring the space between Mediocristan
and Extremistan
Fat Tails are a more general idea than Pareto and Power Law distributions.
One way we can think about it is that “fat-tailedness” is the degree to which
rare events drive the aggregate statistics of a distribution. From this point of
view, fat-tailedness lives on a spectrum from not fat-tailed (i.e. a Gaussian) to
very fat-tailed (i.e. Pareto 80 – 20).
This maps directly to the idea of Mediocristan vs Extremistan discussed
earlier. The image below visualizes different distributions across this
conceptual landscape [2].

print("mean kappa_1n = " + str(np.mean(kappa_dict[filename])))
    print("")
Mean κ (1,100) values from 1000 runs for each dataset. Image by author.
These more stable results indicate Medium followers are the most fat-tailed,
followed by LinkedIn Impressions and YouTube earnings.
Note: One can compare these values to Table III in ref [3] to better understand each
κ value. Namely, these values are comparable to a Pareto distribution with α
between 2 and 3.
Although each heuristic told a slightly different story, all signs point toward
Medium followers gained being the most fat-tailed of the 3 datasets.
Conclusion
While binary labeling data as fat-tailed (or not) may be tempting, fat-
tailedness lives on a spectrum. Here, we broke down 4 heuristics for
quantifying how fat-tailed data are.

Pareto, Power Laws, and Fat Tails
What they don’t teach you in statistics
towardsdatascience.com
Although Pareto (and more generally power law) distributions give us a
salient example of fat tails, this is a more general notion that lives on a
spectrum ranging from thin-tailed (i.e. a Gaussian) to very fat-tailed (i.e.
Pareto 80 – 20).
The spectrum of Fat-tailedness. Image by author.
This view of fat-tailedness provides us with a more flexible and precise way of
categorizing data than simply labeling it as a Power Law (or not). However,
this begs the question: how do we define fat-tailedness?
4 Ways to Quantify Fat Tails

Adding RAG to LLM

We start by downloading the fine-tuned model from the Hugging Face hub.

# load fine-tuned model from hub
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

config = PeftConfig.from_pretrained("shawhin/shawgpt-ft")
model = PeftModel.from_pretrained(model, "shawhin/shawgpt-ft")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

As a baseline, we can see how the model responds to the technical question without any context from the articles. To do this, we create a prompt template using a lambda function, which takes in a viewer comment and returns a prompt for the LLM. For more details on where this prompt comes from, see the previous article of this series.

# prompt (no context)
intstructions_string = f"""ShawGPT, functioning as a virtual data science \
consultant on YouTube, communicates in clear, accessible language, escalating \
to technical depth upon request. It reacts to feedback aptly and ends \
responses with its signature '–ShawGPT'.

ShawGPT will tailor the length of its responses to match the viewer's comment, \
providing concise acknowledgments to brief expressions of gratitude or \
feedback, thus keeping the interaction natural and engaging.

Please respond to the following comment.
"""

prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''
comment = "What is fat-tailedness?"

prompt = prompt_template(comment)
print(prompt)

[INST] ShawGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. It reacts to feedback aptly and ends responses with its signature '–ShawGPT'. ShawGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, thus keeping the interaction natural and engaging.

Please respond to the following comment.
 
What is fat-tailedness? 
[/INST]

Next, we can pass this prompt to the model using the code below.

model.eval()

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), 
                          max_new_tokens=280)

print(tokenizer.batch_decode(outputs)[0])

Here’s the model’s response (no context).

Great question!

Fat-tailedness is a statistical property of a distribution. In simple terms, 
it refers to the presence of extreme outliers or heavy tails in the 
distribution.

For instance, consider the distribution of heights in a population. A normal 
distribution would have most people clustered around an average height with a 
few people deviating slightly from the mean. However, in a fat-tailed 
distribution, you would observe a larger number of people being significantly 
taller or shorter than the average.

In finance, fat-tailedness is often used to describe the presence of extreme 
market moves or outliers in financial returns. It's an important concept in 
risk management and financial modeling.

I hope this explanation helps! Let me know if you have any questions.
–ShawGPT

Although the response's style and formatting are great, the model's explanation of fat-tailedness differs from how I defined it in my video and blog series.

Let’s see what happens to the model’s response when we include the appropriate context. To do this, we create another prompt template, which can also take in context from the RAG system.

# prompt (with context)
prompt_template_w_context = lambda context, comment: f"""[INST]ShawGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature '–ShawGPT'. \
ShawGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
thus keeping the interaction natural and engaging.

{context}
Please respond to the following comment. Use the context above if it is helpful.

{comment}
[/INST]
"""

Next, we pass the prompt with context from the query engine and the view comment to the model.

prompt = prompt_template_w_context(context, comment)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

print(tokenizer.batch_decode(outputs)[0])

Here’s the new response (with context).

Great question!

Fat-tailedness is a property of a distribution that describes how much the 
presence of extreme values (rare events) affects the overall statistics of the 
distribution.

In the context of the video, the author discusses how Mediocristan and 
Extremistan distributions differ in terms of fat-tailedness. Mediocristan 
distributions have a relatively small impact from extreme values, while 
Extremistan distributions are heavily influenced by extreme values.

The author also mentions that fat-tailedness is a more general concept than 
specific distributions like Pareto or Power Laws. Instead, it lives on a 
spectrum ranging from thin-tailed (Gaussian) to very fat-tailed (Pareto 80-20).

I hope that helps clarify things a bit! Let me know if you have any questions.
–ShawGPT

This does a much better job of capturing my explanation of fat tails than the no-context response and even calls out the niche concepts of Mediocristan and Extremistan.
Google Colaboratory
RAG Example Code

colab.research.google.com
What’s next?

Here, I gave a beginner-friendly introduction to RAG and shared a concrete example of how to implement it using LlamaIndex. RAG allows us to improve an LLM system with updateable and domain-specific knowledge.

While much of the recent AI hype has centered around building AI assistants, a powerful (yet less popular) innovation has come from text embeddings (i.e. the things we used to do retrieval). In the next article of this series, I will explore text embeddings in more detail, including how they can be used for semantic search and classification tasks.