from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

from transformers import AutoProcessor


@dataclass
class QAImageOutput:
    q_input_ids: torch.Tensor
    a_input_ids: torch.Tensor
    pixel_values: torch.Tensor


class LlavaDataset(Dataset):
    def __init__(self, dataset_dir: str):
        super().__init__()
        self.chat_data, self.image_dir = self.build_dataest(dataset_dir)

    def build_dataest(self, data_dir: str) -> tuple[list[dict[str, str]], Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("chat.json")
        image_dir = data_dir.joinpath("images")

        chat_data = pd.read_json(chat_file).to_dict(orient='records')

        return chat_data, image_dir

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        q = cur_data['conversations'][0]['value']
        a = cur_data['conversations'][1]['value']
        image_file = self.image_dir.joinpath(cur_data.get('image'))

        return (q, a, image_file)


def build_qaimage(processor: AutoProcessor, q_text: str, a_text: str, image_path: Path):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw_image = Image.open(image_path)

    inputs = processor(images=raw_image, text=prompt, return_tensors="pt")
    a_inputs_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )['input_ids']
    
    return QAImageOutput(
        q_input_ids=inputs.get("input_ids"),
        a_input_ids=a_inputs_ids,
        pixel_values=inputs.get("pixel_values")
    )


"""
q = [101, 102, 103]
a = [201, 202, 203, 204]

input_ids = [101, 102, 103, 201, 202, 203, 204, 'eos_id']
labels = [-100, -100, -100, 201, 202, 203, 204, 'eos_id']
"""

class TrainLlavaModelCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_INDEX: int):
        self.processor = processor
        self.ignore_index = IGNORE_INDEX

    def convert_to_tensor(self,
                          q_input_ids: torch.Tensor,
                          a_input_ids: torch.Tensor):
        # 这里需要注意，图片在process时，有操作如resize，crop等，会把图片缩放到一样的长宽，所以后面collator不需要在对图像进行对齐操作；但是文字不一样，前面处理后，一句话长短不一样
        input_ids = torch.concat([
            q_input_ids,
            a_input_ids,
            torch.tensor([self.processor.tokenizer.eos_token_id]).reshape(1, -1)
        ], dim=1)
        labels = torch.concat([
            torch.full_like(q_input_ids, fill_value=self.ignore_index),
            a_input_ids,
            torch.tensor([self.processor.tokenizer.eos_token_id]).reshape(1, -1)
        ],dim=1)

        return input_ids, labels

    def __call__(self, examples: list[tuple[str, str, Path]]):
        input_ids_list = []
        labels_list = []
        pixel_values_list = []
        max_input_len_list = []

        for example in examples:
            qaimage_output: QAImageOutput = build_qaimage(self.processor, *example)
            tmp_input_ids, tmp_labels = self.convert_to_tensor(qaimage_output.q_input_ids, qaimage_output.a_input_ids)
            max_input_len_list.append(tmp_input_ids.shape[1])
            input_ids_list.append(tmp_input_ids)
            labels_list.append(tmp_labels)
            pixel_values_list.append(qaimage_output.pixel_values)

        max_input_len = max(max_input_len_list)
        
        # 对每个input_ids进行padding到最大长度
        # padded_input_ids = []
        # for idx, input_ids in enumerate(input_ids_list):
        #     current_len = max_input_len_list[idx]
        #     padding_len = max_input_len - current_len
            
        #     if padding_len > 0:
        #         # 创建padding张量
        #         padding = torch.full(
        #             size=(1, padding_len),
        #             fill_value=self.processor.tokenizer.pad_token_id
        #         )
        #         # 拼接padding和原始input_ids
        #         padded_input = torch.concat([padding, input_ids], dim=1)
        #     else:
        #         padded_input = input_ids
            
        #     padded_input_ids.append(padded_input)
        
        # final_input_ids = torch.concat(padded_input_ids, dim=0)

        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(size=(1, max_input_len - max_input_len_list[idx]), fill_value=self.processor.tokenizer.pad_token_id),
                        input_ids
                    ],
                    dim=1
                ) for idx, input_ids in enumerate(input_ids_list) # 左padding
            ],
            dim=0
        )

        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(size=(1, max_input_len - max_input_len_list[idx]), fill_value=self.ignore_index),
                        labels
                    ], dim=1
                ) for idx, labels in enumerate(labels_list) # 左padding
            ], dim=0
        )

        final_pixel_values = torch.concat(pixel_values_list, dim=0)

        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0

        return dict(
            input_ids=final_input_ids,
            attention_mask=attention_mask,
            labels=final_labels,
            pixel_values=final_pixel_values,
        )


if __name__ == "__main__":
    data_dir = "/home/yjp/.cache/huggingface/hub/datasets--liuhaotian--LLaVA-CC3M-Pretrain-595K/snapshots/814894e93db9e12a1dee78b9669e20e8606fd590"
    llavadataset = LlavaDataset(data_dir)
    print(len(llavadataset))
    print(llavadataset[100])