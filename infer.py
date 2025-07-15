import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
root_dir = script_dir / 'VideoLLaMA3'
sys.path.insert(0, str(root_dir))

from VideoLLaMA3.videollama3 import disable_torch_init, model_init, mm_infer
from VideoLLaMA3.videollama3.mm_utils import load_video, load_images, torch


def main():
    disable_torch_init()
    video_path = "examples/UVID_00046.mp4"
    question = (
        "<video>\n"
        "请你分析视频片段，判断其中是否发生了与火灾或其他突发情况相关的应急事件。"
        "请指出各事件其在视频片段中的时间范围，并简要描述事件内容。"
        "若存在多个不同的事件，请按时间顺序输出多个json单元；"
        "当视频中发生新的、有意义的、与火灾相关的事件，或当前事件状态发生显著变化时，"
        "请开始一个新的事件段。事件段可以重叠。\n"
        "每个事件输出要求如下（严格遵循 JSON 格式）：\n"
        "{'emergency_exist': '是' 或 '否',  // 是否发生应急事件\n"
        "  'event_time': '起始秒-结束秒',  // 时间范围，单位为秒，保留一位小数\n"
        "  'event_des': '事件简要描述'  }\n"
        "注意事项：时间是相对于该视频片段的局部时间（即片段起点为 0s）；"
        "所有事件的范围交集要求覆盖全时段。请仅输出符合上述格式的 JSON。"
        "当出现正常、异常事件切换时才区分事件输出，连续相同事件合并为一个事件表述，"
        "事件表述尽可能短。有烟雾也是异常。"
    )

    modal = "video"
    frames, timestamps = load_video(video_path, max_frames=160)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
                {"type": "text", "text": question},
            ]
        }
    ]

    # from peft import PeftModel
    device = "cuda"
    model_path = 'model_ck'

    model, processor = model_init(
        model_path=model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model = model.to(device)

    inputs = processor(
        images=[frames] if modal != "text" else None,
        text=conversation,
        merge_size=2 if modal == "video" else 1,
        return_tensors="pt",
    )

    output = mm_infer(
        inputs,
        model=model,
        tokenizer=processor.tokenizer,
        do_sample=False,
        modal=modal,
        max_new_tokens=180
    )
    print(output)


if __name__ == "__main__":
    main()
