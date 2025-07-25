# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import math
import copy
import json
import os
import pathlib
import random
import re
import sys
import warnings
import traceback
from packaging import version
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
import transformers
from packaging import version
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from peft import PeftModel
sys.path.append('./')

from videollama3.constants import (IGNORE_INDEX,
    NUM_FRAMES, DEFAULT_IMAGE_TOKEN, STREAM_MAX_FRAMES,
    STREAM_START_TOKEN, STREAM_END_TOKEN)
from videollama3.mm_utils import (load_images, load_video,
                                  tokenizer_multimodal_token)
from videollama3.model import *
from videollama3.videollama3_trainer import (
    VideoLLaMA3Trainer, find_all_linear_names, get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer)
from videollama3.model.processor import Videollama3Processor

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None
import gc

from transformers import TrainerCallback

class PeriodicMemoryClearCallback(TrainerCallback):
    def __init__(self, every_n_steps=1):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def int_with_none(value):
    if value == 'None':
        return None
    return int(value)


@dataclass
class ModelArguments:
    # LLM Arguments
    model_type: Optional[str] = field(default="videollama3", metadata={"help": "Model type selected in the list: " + ", ".join(VLLMs.keys())})
    model_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    pretrain_mm_projector: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_encoder: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_attn_implementation: Optional[str] = field(default="flash_attention_2")
    # Token downsampling Arguments
    use_token_compression: Optional[bool] = field(default=False)
    use_flash_loss: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    # Path Arguments
    data_path: List[str] = field(default=None, metadata={"help": "Path to the training data."})
    # image_folder: Optional[str] = field(default=None)
    # video_folder: Optional[str] = field(default=None)
    data_folder: Optional[str] = field(default=None)
    # Loading Arguments
    is_multimodal: bool = False
    fps: Optional[int] = field(default=None)
    max_frames: Optional[int_with_none] = field(default=None)
    # Preprocess Arguments
    image_merge_size: Optional[int] = field(default=1)
    video_merge_size: Optional[int] = field(default=1)
    mm_max_length: Optional[int] = field(default=10240)
    image_aspect_ratio: str = 'square'
    use_batch_flattening: bool = field(default=True, metadata={"help": "Whether to flatten the in-batch sequences of variable lengths."})
    dataset_cache_dir: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # shut auto processing (_remove_unused_columns) of transformers Trainer
    # remove_unused_columns: bool = field(default=False)
    remove_unused_columns=True
    dataloader_persistent_workers: bool = field(default=False, metadata={"help": "Whether to keep dataloader workers persistent between epochs."})
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of subprocesses to use for data loading"}
    )
    
    optim: str = field(default="adamw_torch")
    # Training learning rate Arguments
    vision_encoder_lr: Optional[float] = None
    mm_projector_lr: Optional[float] = None
    llm_lr: Optional[float] = None
    # Training Data Arguments
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 128
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, vlprocessor, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        data_objs = []
        try:
            for data in data_path:
                # NOTE: load_dataset can process both json or jsonl files
                if data.endswith(".json") or data.endswith(".jsonl"):
                    data_objs.append(load_dataset("json", data_files=data, cache_dir=data_args.dataset_cache_dir)["train"])
                else:
                    raise Exception(f"Unsupported file format (<{data}>)!")
            list_data_dict = concatenate_datasets(data_objs)
        except:
            traceback.print_exc()
            # NOTE: compatible with the old version
            list_data_dict = []
            for data in data_path:
                if data.endswith(".json"):
                    data = json.load(open(data, "r"))
                    for i in data:
                        i['id'] = len(list_data_dict)
                        list_data_dict.append(i)
                elif data.endswith(".jsonl"):
                    with open(data, "r", encoding="utf-8") as fp:
                        for line in fp:
                            line = line.strip()
                            obj = json.loads(line)
                            obj["id"] = len(list_data_dict)
                            list_data_dict.append(obj)
                else:
                    raise Exception(f"Unsupported file format (<{data}>)!!!")

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.vlprocessor = vlprocessor
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 576 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def _convert_normal(self, data_dict):
        data_folder = self.data_args.data_folder
        conversation = copy.deepcopy(data_dict["conversations"])

        # data sanity check and repair
        start_idx = 0
        for sentence in conversation:
            if sentence["from"] == "human" or sentence["from"] == "system":
                break
            start_idx += 1
        if start_idx > 0:
            warnings.warn(f"Find {start_idx} non-user sentences at the beginning of the conversation, remove them automatically!")
            conversation = conversation[start_idx:]
        assert len(conversation) > 1, f"Invalid conversation"

        if 'image' in data_dict and data_dict['image'] is not None:
            modal = 'image'
            if all(not "<image>" in sentence["value"] for sentence in conversation):
                warnings.warn(f"Image tag not found in the conversation, add it automatically at the beginning!")
                conversation[0]["value"] = "<image>" + conversation[0]["value"]
            image_file = data_dict['image']
            if isinstance(image_file, list):
                image_file = [os.path.join(data_folder, f) for f in image_file]
            else:
                image_file = os.path.join(data_folder, image_file)
            images = load_images(image_file)
        elif 'video' in data_dict and data_dict['video'] is not None:
            modal = 'video'
            if all(not "<video>" in sentence["value"] for sentence in conversation):
                warnings.warn(f"Video tag not found in the conversation, add it automatically at the beginning!")
                conversation[0]["value"] = "<video>" + conversation[0]["value"]
            video_file = data_dict['video']
            if isinstance(video_file, list) and len(video_file) == 1:
                video_file = os.path.join(data_folder, video_file[0])
                images, timestamps = load_video(video_file, fps=self.data_args.fps, max_frames=self.data_args.max_frames)
                images = [images]
            else:
                raise ValueError(f"Unsupported video format: {video_file}")
        else:
            modal = 'text'
            images = None

        messages = []
        for conv in conversation:
            if conv["from"] == "human":
                # replace video tag to image tag for unified processing
                # conv["value"] = conv["value"].replace("<video>", "<image>" * len(images))
                chunks = conv["value"].split("<image>" if modal == 'image' else "<video>")
                messages.append({
                    "role": "user",
                    "content": []
                })

                for chunk_idx in range(1, 2 * len(chunks)):
                    if chunk_idx % 2 == 1:
                        chunk = chunks[chunk_idx // 2].strip()
                        messages[-1]["content"].append({"type": "text",  "text": chunk}) if chunk else None
                    else:
                        if modal == 'image':
                            messages[-1]["content"].append({"type": "image"})
                        elif modal == 'video':
                            messages[-1]["content"].append({"type": "video", "num_frames": len(images[0]), "timestamps": timestamps})
            else:
                messages.append({
                    "role": "assistant",
                    "content": conv['value']
                })

        if modal == 'video':
            merge_size = self.data_args.video_merge_size
        else:
            # image/text
            merge_size = self.data_args.image_merge_size

        return modal, images, messages, merge_size

    def _convert_stream(self, data_dict):
        video_path = os.path.join(self.data_args.data_folder, data_dict['video'][0])
        frames, timestamps = load_video(
            video_path=video_path,
            start_time=data_dict["start_time"],
            end_time=data_dict["end_time"],
            fps=self.data_args.fps,
            max_frames=self.data_args.max_frames,
        )

        if len(frames) > STREAM_MAX_FRAMES:
            max_time = timestamps[STREAM_MAX_FRAMES]
            frames = frames[:STREAM_MAX_FRAMES]
            timestamps = timestamps[:STREAM_MAX_FRAMES]
        else:
            max_time = float("inf")

        messages = []
        frame_idx = 0

        conversation = copy.deepcopy(data_dict["conversation"])
        for message in conversation:
            if message["time"] >= max_time:
                break

            while frame_idx < len(timestamps) and timestamps[frame_idx] <= message["time"]:
                messages.append({
                    "role": "stream",
                    "content": [{"type": "image", "timestamps": timestamps[frame_idx] - data_dict["start_time"]}],
                })
                frame_idx += 1

            messages.append(message)

        frames = frames[:frame_idx]

        return "video", [frames], messages, self.data_args.video_merge_size

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_dict = self.list_data_dict[i]

        try:
            if "stream" in data_dict and data_dict["stream"]:
                modal, images, messages, merge_size = self._convert_stream(data_dict)
            else:
                modal, images, messages, merge_size = self._convert_normal(data_dict)

            data_dict = self.vlprocessor(
                images=images,
                text=messages,
                merge_size=merge_size,
                return_labels=True,
                return_tensors="pt",
            )

            if modal == 'text':
                unit_size = self.vlprocessor.image_processor.patch_size**2 * 3
                data_dict['pixel_values'] = torch.zeros(self.data_args.image_merge_size**2, unit_size)
                data_dict['grid_sizes'] = torch.as_tensor([[1, self.data_args.image_merge_size, self.data_args.image_merge_size]])
                data_dict['merge_sizes'] = torch.as_tensor([self.data_args.image_merge_size])
            elif modal == 'image' or modal == 'video':
                assert len(data_dict['pixel_values']) > 0 and len(data_dict['grid_sizes']) > 0, f"Invalid image data: {data_dict['images']}, {data_dict['grid_thws']}"

            data_dict['modals'] = [modal] * len(images)

        except Exception as e:
            traceback.print_exc()
            backup_idx = random.randint(0, len(self.list_data_dict) - 1)
            print(f"Encounted error when process {i}-th example: {data_dict}, use {backup_idx}-th example instead!!!")
            return self.__getitem__(backup_idx)

        del images  # ← 释放原始帧
        del messages
        torch.cuda.empty_cache()
        gc.collect()
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    vlprocessor: transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.vlprocessor.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.vlprocessor.tokenizer.model_max_length]
        labels = labels[:, :self.vlprocessor.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.vlprocessor.tokenizer.pad_token_id),
        )

        # work for 'images' argument in `prepare_inputs_labels_for_multimodal`
        batch["pixel_values"] = torch.cat([x["pixel_values"] for x in instances])
        batch["grid_sizes"] = torch.cat([x["grid_sizes"] for x in instances])
        batch["merge_sizes"] = torch.cat([x["merge_sizes"] for x in instances])
        batch["modals"] = sum([x["modals"] for x in instances], [])

        return batch


def make_supervised_data_module(vlprocessor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        vlprocessor=vlprocessor,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(vlprocessor=vlprocessor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


@dataclass
class DataCollatorWithFlatteningForSupervisedDataset(object):
    """Collate examples for batch flattened supervised fine-tuning."""

    vlprocessor: transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict], separator_id=-100) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        new_input_ids = []
        new_labels = []
        position_ids = []
        for idx in range(0, len(input_ids)):
            new_input_ids.append(input_ids[idx][:self.vlprocessor.tokenizer.model_max_length])
            temp_label = labels[idx][:self.vlprocessor.tokenizer.model_max_length]
            temp_label[0] = separator_id
            new_labels.append(temp_label)
            position_ids.append(torch.tensor(list(range(len(input_ids[idx][:self.vlprocessor.tokenizer.model_max_length])))))

        new_input_ids = torch.cat(new_input_ids)
        new_labels = torch.cat(new_labels)
        position_ids = torch.cat(position_ids)

        batch = dict(
            input_ids=new_input_ids.unsqueeze(0),
            labels=new_labels.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
        )

        # work for 'images' argument in `prepare_inputs_labels_for_multimodal`
        batch["pixel_values"] = torch.cat([x["pixel_values"] for x in instances])
        batch["grid_sizes"] = torch.cat([x["grid_sizes"] for x in instances])
        batch["merge_sizes"] = torch.cat([x["merge_sizes"] for x in instances])
        batch["modals"] = sum([x["modals"] for x in instances], [])

        return batch


def make_flattening_supervised_data_module(vlprocessor: transformers.ProcessorMixin, data_args) -> Dict:
    """Make batch flattened dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        vlprocessor=vlprocessor,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorWithFlatteningForSupervisedDataset(vlprocessor=vlprocessor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    if local_rank == 0:
        print('------model args------')
        print(model_args)
        print('------data args------')
        print(data_args)
        print('------training args------')
        print(training_args)

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    model_args.torch_dtype = compute_dtype

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},
            # BUG: High version transformers report error:
            # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))

    config = VLLMConfigs[model_args.model_type].from_pretrained(model_args.model_path)

    config._attn_implementation = attn_implementation
    config.use_token_compression = model_args.use_token_compression
    config.use_flash_loss = model_args.use_flash_loss

    if model_args.vision_encoder is not None:
        config.vision_encoder = model_args.vision_encoder
        model = VLLMs[model_args.model_type].from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype=compute_dtype,
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
        if 'mixtral' in model_args.model_type:
            import deepspeed
            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype=compute_dtype,
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        '''
        from peft import PeftModel
        print('Loading LoRA weights...')
        model_path = '/data/zhouting/outputs/fire_analysis/models/videollama3_7b_local_new_2stage/ft_stageI_batch32_180f_2e'
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
        # 设置为训练模式
        model.train()

        # 冻结除LoRA参数外的所有参数
        for name, param in model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
        '''
        

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.vision_encoder is not None:
        # initialize vision encoder + multi-modal projector
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_encoder = model.get_vision_encoder()
        vision_encoder.to(dtype=compute_dtype, device=training_args.device)

        mm_max_length = data_args.mm_max_length
        vision_encoder.image_processor.max_tokens = mm_max_length

        mm_projector = model.get_mm_projector()
        mm_projector.to(dtype=compute_dtype if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.is_multimodal = True

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        # decoupled learning rate
        model.config.llm_lr = training_args.llm_lr
        model.config.vision_encoder_lr = training_args.vision_encoder_lr
        model.config.mm_projector_lr = training_args.mm_projector_lr

        if model.config.llm_lr is None:
            for p in model.get_model().parameters():
                p.requires_grad = False
            for p in model.get_model().vision_encoder.parameters():
                p.requires_grad = True
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        if model.config.vision_encoder_lr is None:
            for p in model.get_model().vision_encoder.parameters():
                p.requires_grad = False

        if model.config.mm_projector_lr is None:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.max_frames = getattr(data_args, 'max_frames', NUM_FRAMES)
        model.config.image_aspect_ratio = data_args.image_aspect_ratio if 'avt' not in model_args.vision_encoder else 'avt'

        # NOTE: complement data_args via model hyperparameters
        # 1. acquire image size
        model.config.image_size = data_args.image_size = vision_encoder.image_size
        # 2. calculate the number of tokens in the image
        model.config.image_token_length = data_args.image_token_length = mm_projector.cal_proj_size(vision_encoder.num_patches_per_side)
        # 3. check if alignment
        model.config.is_alignment = training_args.is_alignment = data_args.is_alignment = (
            model.config.mm_projector_lr is not None and
            model.config.llm_lr is None and
            model.config.vision_encoder_lr is None
        )
        # 4. set spatial merge size as default
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN, STREAM_START_TOKEN, STREAM_END_TOKEN], special_tokens=True)
        model.config.image_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)

        vlprocessor = Videollama3Processor(vision_encoder.image_processor, tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if local_rank == 0:
        print("Current model:", model)
        print("Model config:", model.config)

    if data_args.use_batch_flattening:
        rank0_print('You are using flattening operation to flatten the entire mini batch into a single sequence')
        assert model.config._attn_implementation == 'flash_attention_2'
        assert version.parse(transformers.__version__) >= version.parse("4.44.0")
        data_module = make_flattening_supervised_data_module(vlprocessor=vlprocessor, data_args=data_args)
    else:
        data_module = make_supervised_data_module(vlprocessor=vlprocessor, data_args=data_args)

    # select a Trainer
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"\n\n\n\n\n Trainable parameters: {len(trainable)}")
    # 总参数数量
    total_params = sum(p.numel() for p in model.parameters())

    # 训练参数数量（requires_grad=True）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 百分比
    trainable_percent = 100 * trainable_params / total_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {trainable_percent:.2f}%")
    print('\n\n\n\n\n\n')
    trainer = VideoLLaMA3Trainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=[PeriodicMemoryClearCallback(every_n_steps=1)], **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
