from torch.utils.data import Dataset
from datasets import load_dataset
import transformers
from typing import Dict
import torch
# import numpy as np
# from tqdm import tqdm
import json
import random
import csv
import numpy as np
import pandas as pd

random.seed(42)

class SafetyAlignDataset(Dataset):
    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer,
                model_name_or_path,
                hybrid_response=True,
                num_augmentations=1
            ):
        super(SafetyAlignDataset, self).__init__()
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path.lower()
        self.max_length = 1024
        self.benign_num_examples = 4000
        self.malicious_num_examples = 2000
        
        if hybrid_response and num_augmentations>0:
            self.malicious_num_examples += self.malicious_num_examples*num_augmentations
    
        one_shot_template = None
        user_tag, assistant_tag, dialog_end_token = None, None, None
        sep_token = ""

        switch_select = [0]
        if 'llama3' in self.model_name_or_path or 'llama-3' in self.model_name_or_path:
            one_shot_template = "{user_tag}{instruction}{dialog_end_token}{assistant_tag}<SEPARATOR>{response}{dialog_end_token}"
            print("use LLAMA-3 template")
            user_tag="<|start_header_id|>user<|end_header_id|>\n\n"
            assistant_tag="<|start_header_id|>assistant<|end_header_id|>\n\n"
            dialog_end_token = "<|eot_id|>"
            switch_select = [0, 1]
        elif 'llama2' in self.model_name_or_path or 'llama-2' in self.model_name_or_path:
            print("use LLAMA-2 template")
            one_shot_template = "<s>{user_tag}{dialog_end_token}{instruction}{dialog_end_token}{assistant_tag}{dialog_end_token}<SEPARATOR>{response}{dialog_end_token}</s>"
            user_tag="[INST] "
            assistant_tag="[/INST] "
            dialog_end_token = " "
            switch_select = [0, 1]
        elif 'gemma' in self.model_name_or_path:
            print("use GEMMA template")
            one_shot_template = "<bos>{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}{dialog_end_token}"
            user_tag="<start_of_turn>user\n"
            assistant_tag="<end_of_turn>\n<start_of_turn>model\n"
            dialog_end_token = "<eos>"
            switch_select = [0, 1]
        elif 'mistral' in self.model_name_or_path:
            print("use MISTRAL template")
            sep_token = " "
            one_shot_template = "<s>{user_tag}{dialog_end_token}{instruction}{dialog_end_token}{assistant_tag}{dialog_end_token}<SEPARATOR>{response}{dialog_end_token}</s>"
            user_tag="[INST] "
            assistant_tag="[/INST] "
            dialog_end_token = ""
            switch_select = [0]
        else:
            raise NotImplementedError(f"Unsupported model type: {self.model_name_or_path}")
        
        assert user_tag and assistant_tag, "user_tag/assistant_tag undefined"

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.sep_token = sep_token

        benign_train_data = []

        source_data = pd.read_csv('dataset/overrefusal_mitigate.csv')
        source_data = source_data[['prompt','generated_response']]
        for i, d in source_data.iterrows():  
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag, dialog_end_token=dialog_end_token,
                    instruction=d['prompt'], response=d['generated_response'])
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag, dialog_end_token=dialog_end_token,
                    instruction="", response=d['generated_response'])
            benign_train_data.append(formatted_input)

        benign_train_data *= 2 # Augment

        ds_path = "dataset/ultrachat_200k-test_sft.arrow"
        ds = load_dataset("arrow", data_files=ds_path)['train']
        for example in ds:
            messages = example["messages"]
            if len(messages) > 2: messages = messages[:2]

            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag, dialog_end_token=dialog_end_token,
                    instruction=messages[0]['content'], response=messages[1]["content"])
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag, dialog_end_token=dialog_end_token,
                    instruction="", response=messages[1]["content"])
            benign_train_data.append(formatted_input)

            if len(benign_train_data) >= self.benign_num_examples:
                break

        malicious_train_data = []
        with open('./data/train/circuit_breakers_train_processed_2k.json', 'r') as f:
            circuit_breaker_data = json.load(f)

        for data in circuit_breaker_data:
            query = data['prompt']
            refusal_response = data['refusal_answer']
            malicious_response = data['output']

            formatted_input = one_shot_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag, dialog_end_token=dialog_end_token,
                instruction=query, response=refusal_response)
            malicious_train_data.append(formatted_input)

            if hybrid_response:
                for _ in range(num_augmentations):
                    tokens = self.tokenizer.tokenize(malicious_response)
                    # Changing the parameters here may achieve better results.
                    split_idx = random.randint(20, 25)
                    prefix_tokens = tokens[:split_idx]
                    prefix_harmful_text = self.tokenizer.convert_tokens_to_string(prefix_tokens)

                    if split_idx > 0:
                        final_reponse = "[PH]" + prefix_harmful_text + "[/PH]" + refusal_response
                    else:
                        final_reponse = prefix_harmful_text + refusal_response

                    formatted_input = one_shot_template.format(
                        user_tag=user_tag, assistant_tag=assistant_tag, dialog_end_token=dialog_end_token,
                        instruction=query, response=final_reponse)
                    malicious_train_data.append(formatted_input)

            if len(malicious_train_data) >= self.malicious_num_examples:
                break

        combined_data = [(sample, True) for sample in benign_train_data] + [(sample, False) for sample in malicious_train_data]
        random.shuffle(combined_data)
        self.train_data, self.is_benign = zip(*combined_data)
        self.train_data = list(self.train_data)
        self.is_benign = list(self.is_benign)
        print("[*] total samples:", len(self.train_data))
        print("[*] train example:", self.train_data[0])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        train_text = self.train_data[i]
        is_benign = self.is_benign[i]

        query, hybrid_reponse = train_text.split('<SEPARATOR>')
        train_text = train_text.replace('<SEPARATOR>', '').replace("[PH]", "").replace("[/PH]", "")

        prefix_len = 0
        if "[PH]" in hybrid_reponse and "[/PH]" in hybrid_reponse:
            prefix_harmful_text = hybrid_reponse.split("[PH]")[1].split("[/PH]")[0]
            assert prefix_harmful_text != "", "prefix_harmful_text is empty"
            prefix_tokens = self.tokenizer.tokenize(prefix_harmful_text)
            prefix_len = len(prefix_tokens)

        tokenize_kwargs = dict(
            max_length=self.max_length, 
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        self.tokenizer.padding_side = "left"
        train_tokens = self.tokenizer(
            train_text,
            **tokenize_kwargs
        )
        input_ids = train_tokens["input_ids"]
        attention_mask = train_tokens["attention_mask"]
        labels = input_ids.clone()

        query_tokens = self.tokenizer(
            query, 
            max_length=self.max_length, 
            truncation=True, 
            add_special_tokens=True
        )
        query_len = len(query_tokens["input_ids"])
        start_index = attention_mask[0].nonzero(as_tuple=True)[0][0]
        labels[:, start_index : start_index + query_len + prefix_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            is_benign=torch.tensor(is_benign),
            response_start_idx=torch.tensor(query_len),
        )
