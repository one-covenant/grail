# %% [markdown]
# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
#
# To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://docs.unsloth.ai/get-started/installing-+-updating).
#
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
#

# %% [markdown]
# ### News

# %% [markdown]
#
# Unsloth now supports [gpt-oss RL](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning) with the fastest inference & lowest VRAM. Try our [new notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb) which automatically creates kernels!
#
# [Vision RL](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) is now supported! Train Qwen2.5-VL, Gemma 3 etc. with GSPO or GRPO.
#
# Introducing Unsloth [Standby for RL](https://docs.unsloth.ai/basics/memory-efficient-rl): GRPO is now faster, uses 30% less memory with 2x longer context.
#
# Unsloth now supports Text-to-Speech (TTS) models. Read our [guide here](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning).
#
# Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks).
#

# %% [markdown]
# ### Unsloth

# %% [markdown]
# Goal: To convert `Qwen3-4B-Base` into a reasoning model via GRPO by using OpenR1's Math dataset.
#
# We first pre fine-tune the model to make GRPO skip trying to match formatting - this speeds GRPO up.

# %%
import gc
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from vllm import SamplingParams

max_seq_length = 2048  # Can increase for longer reasoning traces
lora_rank = 32  # Larger rank = smarter, but slower

SAVE_BASE_DIR = Path("/ephemeral")
SFT_OUTPUT_DIR = SAVE_BASE_DIR / "outputs_ft_sft"
GRPO_OUTPUT_DIR = SAVE_BASE_DIR / "outputs_ft"
FULL_MODEL_OUTPUT_DIR = SAVE_BASE_DIR / "grpo_saved_model_ft"
MERGED_32_DIR = SAVE_BASE_DIR / "model_ft_32bit"
MERGED_16_DIR = SAVE_BASE_DIR / "model_ft_16bit"
MERGED_4_DIR = SAVE_BASE_DIR / "model_ft_4bit"

for directory in (
    SFT_OUTPUT_DIR,
    GRPO_OUTPUT_DIR,
    FULL_MODEL_OUTPUT_DIR,
    MERGED_32_DIR,
    MERGED_16_DIR,
    MERGED_4_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# Latest Unsloth builds require this flag for full-parameter training
os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "1"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=max_seq_length,
    load_in_4bit=False,  # False for full fine-tuning in 16bit
    fast_inference=True,  # Enable vLLM fast inference
    gpu_memory_utilization=0.7,  # Reduced for full fine-tuning to avoid OOM
)

model.requires_grad_(True)

# For full fine-tuning, Unsloth skips LoRA adapters when this env var is set
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing="unsloth",  # Critical for memory savings
    random_state=3407,
)

# %% [markdown]
# ### GRPO chat template
# Since we're using a base model, we should set a chat template. You can make your own chat template as well!
# 1. DeepSeek uses `<think>` and `</think>`, but this is **not** necessary - you can customize it however you like!
# 2. A `system_prompt` is recommended to at least guide the model's responses.

# %%
reasoning_start = "<start_working_out>"  # Acts as <think>
reasoning_end = "<end_working_out>"  # Acts as </think>
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
# system_prompt

# %% [markdown]
# We create a simple chat template below. Notice `add_generation_prompt` includes prepending `<start_working_out>` to guide the model to start its reasoning process.

# %%
chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ '{system_prompt}' + eos_token }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
    "{% endif %}"
)

# Replace with out specific template:
chat_template = chat_template.replace("'{system_prompt}'", f"'{system_prompt}'").replace(
    "'{reasoning_start}'", f"'{reasoning_start}'"
)
tokenizer.chat_template = chat_template

# %% [markdown]
# Let's see how our chat template behaves on an example:

# %%
tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "What is 1+1?"},
        {
            "role": "assistant",
            "content": f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}",
        },
        {"role": "user", "content": "What is 2+2?"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

# %% [markdown]
# ### Pre fine-tuning for formatting
# We now use a subset of NVIDIA's [Open Math Reasoning dataset](https://huggingface.co/datasets/nvidia/OpenMathReasoning) which was filtered to only include high quality DeepSeek R1 traces.
#
# We'll only filter ~59 or so examples to first "prime" / pre fine-tune the model to understand our custom GRPO formatting.

# %%

dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
dataset = dataset.to_pandas()[["expected_answer", "problem", "generated_solution"]]

# Try converting to number - if not, replace with NaN
is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors="coerce").notnull()
# Select only numbers
dataset = dataset.iloc[np.where(is_number)[0]]

# dataset

# %% [markdown]
# We have to format the dataset to follow our GRPO style formatting:


# %%
def format_dataset(x):
    expected_answer = x["expected_answer"]
    problem = x["problem"]

    # Remove generated <think> and </think>
    thoughts = x["generated_solution"]
    thoughts = thoughts.replace("<think>", "").replace("</think>", "")

    # Strip newlines on left and right
    thoughts = thoughts.strip()
    # Add our custom formatting
    final_prompt = (
        reasoning_start + thoughts + reasoning_end + solution_start + expected_answer + solution_end
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": final_prompt},
    ]


dataset["Messages"] = dataset.apply(format_dataset, axis=1)

# %% [markdown]
# Check to see if it worked:

# %%
tokenizer.apply_chat_template(dataset["Messages"][0], tokenize=False)

# %% [markdown]
# Let's truncate the pre fine-tuning dataset to `max_seq_length/2` since we don't want too long reasoning traces.
#
# Note this might take 2 minutes!

# %%
dataset["N"] = dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))

dataset = dataset.loc[dataset["N"] <= max_seq_length / 2].copy()
# dataset.shape

# %% [markdown]
# We then tokenize the messages and convert it to a Hugging Face compatible dataset format:

# %%
dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize=False)
dataset = Dataset.from_pandas(dataset)
# dataset

# %% [markdown]
# Let's now pre fine-tune the model so it follows our custom GRPO formatting!

# %%
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Increased to reduce memory usage
        warmup_steps=5,
        num_train_epochs=2,  # Set this for 1 full training run.
        learning_rate=5e-6,  # Lower LR for full fine-tuning (was 2e-4 for LoRA)
        logging_steps=5,
        optim="adamw_8bit",  # 8bit optimizer to save memory
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",  # Use this for WandB etc
        output_dir=str(SFT_OUTPUT_DIR),  # Different directory for full fine-tuning SFT
        save_steps=50,
        save_total_limit=2,
    ),
)

# %%
trainer.train()

# %% [markdown]
# Let's check if the model has learnt to follow the custom format:

# %%
text = tokenizer.apply_chat_template(
    dataset[0]["Messages"][:2],
    tokenize=False,
    add_generation_prompt=True,  # Must add for generation
)
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    temperature=0,
    max_new_tokens=1024,
    streamer=TextStreamer(tokenizer, skip_prompt=False),
)

# %% [markdown]
# Yes it did follow the formatting! Great! Let's remove some items before the GRPO step

# %%
del dataset
torch.cuda.empty_cache()

gc.collect()

# %% [markdown]
# ### Data Prep
# <a name="Data"></a>
#
# We're using Hugging Face's [Open R1 Math dataset](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed). You can also utilize OpenAI's famous [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k)

# %%
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
# dataset

# %% [markdown]
# Let's look at the first row:

# %%
dataset[0]["prompt"]

# %%
dataset[0]["solution"]

# %% [markdown]
# In GSM8K, ee notice all answers like about have a ####, so we extract it. But for the Open R1 dataset, we can skip the below.


# %%
def extract_hash_answer(text):
    # if "####" not in text: return None
    # return text.split("####")[1].strip()
    return text


extract_hash_answer(dataset[0]["solution"])

# %% [markdown]
# Let's map the dataset! and see the first row:

# %%
dataset = dataset.map(
    lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    }
)
dataset[0]

# %% [markdown]
# We create a regex format to match the reasoning sections and answers:

# %%

# Add optional EOS token matching
solution_end_regex = r"</SOLUTION>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"

match_format = re.compile(
    rf"{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end_regex}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
# match_format

# %% [markdown]
# We verify it works:

# %%
match_format.findall(
    "Let me think!<end_working_out><SOLUTION>\n2\n</SOLUTION>",
)

# %%
match_format.findall(
    "<start_working_out>Let me think!<end_working_out><SOLUTION>  2  </SOLUTION>\n\n",
)

# %% [markdown]
# We now want to create a reward function to match the format exactly - we reward it with 3 points if it succeeds:


# %%
def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


# %% [markdown]
# If it fails, we want to reward the model if it at least follows the format partially, by counting each symbol:


# %%
def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!

        # No need to reward <start_working_out> since we always prepend it!
        # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores


# %% [markdown]
# Finally, we want to extract the generated answer, and reward or penalize it! We also reward it based on how close the answer is to the true one via ratios:


# %%
def check_answer(prompts, completions, answer, **kwargs):
    # question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer, strict=False):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        # Correct answer gets 5 points!
        if guess == true_answer:
            score += 5.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 1.5
                else:
                    score -= 2.5  # Penalize wrong answers
            except Exception:
                score -= 4.5  # Penalize
        scores.append(score)
    return scores


# %% [markdown]
# Also sometimes it might not be 1 number as the answer, but like a sentence for example "The solution is $20" -> we extract 20.
#
# We also remove possible commas for example as in 123,456

# %%
match_numbers = re.compile(
    solution_start + r".*?[\s]{0,}([-]?[\d\.,]{1,})",
    flags=re.MULTILINE | re.DOTALL,
)
print(match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>"))
print(match_numbers.findall("<SOLUTION>  123,456  </SOLUTION>"))
print(match_numbers.findall("<SOLUTION>  -0.234  </SOLUTION>"))
print(match_numbers.findall("<SOLUTION>17</SOLUTION>"))

# %% [markdown]
# We now prepare our main function which will print out the generated responses and the true answer, along with another reward function which converts text to float via `float` and sees if it's the same.

# %%
global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5


def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    # Print only every few steps
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            "*" * 20 + f"Question:\n{question}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
    PRINTED_TIMES += 1

    for guess, true_answer in zip(extracted_responses, answer, strict=False):
        if guess is None:
            scores.append(-2.5)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            # Remove commas like in 123,456
            guess = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess == true_answer else -1.5)
        except Exception:
            scores.append(0)
            continue
    return scores


# %% [markdown]
# Get the top 90% prompt length so we don't accidentally truncate them!
#
# Ie we'll remove the top 10% long prompts.

# %%
tokenized = dataset.map(
    lambda x: {
        "tokens": tokenizer.apply_chat_template(
            x["prompt"], add_generation_prompt=True, tokenize=True
        )
    },
    batched=True,
)
print(tokenizer.decode(tokenized[0]["tokens"]))
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length = ", maximum_length)

# Filter only samples smaller than 90% max length
dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized

# %% [markdown]
# <a name="Train"></a>
# ### Train the model
#
# Now set up GRPO Trainer and all configurations!

# %%
max_prompt_length = maximum_length + 1  # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=1e-6,  # Lower LR for full fine-tuning (was 5e-6 for LoRA)
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",  # 8bit optimizer critical for memory savings
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Increased to reduce memory footprint
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    beta=0.0,
    max_steps=100,
    save_steps=1,
    save_total_limit=11,
    report_to="none",
    output_dir=str(GRPO_OUTPUT_DIR),
    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)

# %% [markdown]
# And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!
#
# You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!
#
# | Step | Training Loss | reward    | reward_std | completion_length | kl       |
# |------|---------------|-----------|------------|-------------------|----------|
# | 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
# | 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
# | 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |
#

# %%
# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args=training_args,
    train_dataset=dataset,
    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)
trainer.train()

# %% [markdown]
# <a name="Inference"></a>
# ### Inference
# Now let's try the model we just trained! First, let's first try the model without any GRPO trained:

# %%
text = "What is the sqrt of 101?"

sampling_params = SamplingParams(
    temperature=1.0,
    top_k=50,
    max_tokens=1024,
)
output = (
    model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0]
    .outputs[0]
    .text
)

# output

# %% [markdown]
# For full fine-tuning, we save the entire model instead of just LoRA adapters

# %%
# Save the full fine-tuned model
model.save_pretrained(str(FULL_MODEL_OUTPUT_DIR))
tokenizer.save_pretrained(str(FULL_MODEL_OUTPUT_DIR))

# %% [markdown]
# Verify model weights have changed (for full fine-tuning)

# %%
# For full fine-tuning, we can verify the model was saved correctly

assert (FULL_MODEL_OUTPUT_DIR / "config.json").exists(), "Model config not found"
assert (FULL_MODEL_OUTPUT_DIR / "model.safetensors").exists() or (
    FULL_MODEL_OUTPUT_DIR / "pytorch_model.bin"
).exists(), "Model weights not found"
print("✅ Full fine-tuned model saved successfully!")

# %% [markdown]
# Now we load the full model and test:

# %%
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the sqrt of 101?"},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,  # Must add for generation
    tokenize=False,
)

sampling_params = SamplingParams(
    temperature=1.0,
    top_k=50,
    max_tokens=2048,
)
# For full fine-tuning, no need to load LoRA - the model itself is already fine-tuned
output = (
    model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=None,  # No LoRA needed for full fine-tuning
    )[0]
    .outputs[0]
    .text
)

# output

# %% [markdown]
# Our reasoning model is much better - it's not always correct, since we only trained it for an hour or so - it'll be better if we extend the sequence length and train for longer!

# %% [markdown]
# <a name="Save"></a>
# ### Saving to float16 for VLLM
#
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# %%
# For full fine-tuning, save the entire model (no merging needed)
# Save in 32bit for maximum precision
if True:
    model.save_pretrained_merged(
        str(MERGED_32_DIR),
        tokenizer,
        save_method="merged_32bit",
    )

# Also save 16bit version for deployment
if True:
    model.save_pretrained_merged(
        str(MERGED_16_DIR),
        tokenizer,
        save_method="merged_16bit",
    )

# Push to hub (optional)
if False:
    model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_16bit", token="")

# Merge to 4bit (optional, for edge deployment)
if False:
    model.save_pretrained_merged(
        str(MERGED_4_DIR),
        tokenizer,
        save_method="merged_4bit",
    )


# %% [markdown]
# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
#
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
#
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)

# %%
# Save to 8bit Q8_0
if False:
    model.save_pretrained_gguf(
        "model",
        tokenizer,
    )
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, token="")

# Save to 16bit GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="f16", token="")

# Save to q4_k_m GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="q4_k_m", token="")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model",  # Change hf to your username!
        tokenizer,
        quantization_method=[
            "q4_k_m",
            "q8_0",
            "q5_k_m",
        ],
        token="",
    )

# %% [markdown]
# Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp.
#
# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
#
# Some other links:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!
#
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
#
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
#
