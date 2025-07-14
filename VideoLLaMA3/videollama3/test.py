import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
import numpy as np
from tqdm import tqdm
from typing import Callable, List, Dict

# 设置项目根目录
os.chdir('/home/zhouting/SurveillanceVideoFireAnalysis/VideoLLaMA3')
sys.path.append('./')

from videollama3 import disable_torch_init, model_init, mm_infer
from videollama3.mm_utils import load_video, torch


def compute_iou(gt_spans: List[List[float]], pred_spans: List[List[float]]) -> float:
    def merge_spans(spans):
        if not spans:
            return []
        spans = sorted(spans)
        merged = [spans[0]]
        for s, e in spans[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged

    def span_length(spans):
        return sum(e - s for s, e in spans)

    def intersection(s1, s2):
        result = []
        for a, b in s1:
            for c, d in s2:
                start = max(a, c)
                end = min(b, d)
                if start < end:
                    result.append([start, end])
        return result

    gt_merged = merge_spans(gt_spans)
    pred_merged = merge_spans(pred_spans)
    inter = intersection(gt_merged, pred_merged)

    inter_len = span_length(inter)
    union_len = span_length(gt_merged) + span_length(pred_merged) - inter_len

    return inter_len / union_len if union_len > 0 else 1.0

import re
def parse_event_time(event_time_str: str):
    """
    尝试从 event_time 字符串中提取两个时间点（秒数），返回 [start, end]，否则返回 None。
    支持形式：'12.0s-16.0s'，'12秒-16秒'，'12.0s 到 16.0秒'，'12-16s'，'12.0 second - 16.0 second' 等
    """
    # 将所有中文、英文、空格单位统一替换为 s
    s = event_time_str.replace("秒", "s").replace("second", "s").replace(" ", "").replace("到", "-")

    # 正则提取两个时间点
    matches = re.findall(r'(\d+(?:\.\d+)?)s?', s)
    if len(matches) >= 2:
        try:
            start = float(matches[0])
            end = float(matches[1])
            return [start, end] if start <= end else [end, start]
        except ValueError:
            return None
    
    
    return None

def evaluate_model_on_dataset(
    dataset: List[Dict],
    predict_fn: Callable[[Dict], str]
) -> Dict[str, float]:
    total = len(dataset)
    tp = fp = fn = valid = 0
    pos_ious = []
    neg_ious = []
    all_ious = []
    start_time_diffs = []  # 记录所有有对齐异常的起始时刻差

    for item in tqdm(dataset, desc="Evaluating"):
        gt_str = item['conversations'][-1]['value']
        try:
            gt_events = json.loads(gt_str.replace("'", '"'))
        except Exception:
            continue

        # ground-truth spans
        gt_pos_spans = [
            [float(t.replace("s", "")) for t in e["event_time"].split("-")]
            for e in gt_events if e.get("emergency_exist") == "是"
        ]
        gt_neg_spans = [
            [float(t.replace("s", "")) for t in e["event_time"].split("-")]
            for e in gt_events if e.get("emergency_exist") == "否"
        ]
        gt_all_spans = gt_pos_spans + gt_neg_spans

        gt_emergency = len(gt_pos_spans) > 0

        try:
            pred_str = predict_fn(item)
            pred_events = json.loads(pred_str.replace("'", '"'))
            valid += 1
        except Exception:
            pred_events = []

        pred_pos_spans = [
            parse_event_time(e["event_time"])
            for e in pred_events if e.get("emergency_exist") == "是"
        ]
        pred_neg_spans = [
            parse_event_time(e["event_time"])
            for e in pred_events if e.get("emergency_exist") == "否"
        ]

        # 去掉解析失败的
        pred_pos_spans = [span for span in pred_pos_spans if span]
        pred_neg_spans = [span for span in pred_neg_spans if span]

        pred_all_spans = pred_pos_spans + pred_neg_spans

        pred_emergency = len(pred_pos_spans) > 0

        # classification-level
        if gt_emergency and pred_emergency:
            tp += 1
        elif not gt_emergency and pred_emergency:
            fp += 1
        elif gt_emergency and not pred_emergency:
            fn += 1

        # IoUs
        if gt_pos_spans or pred_pos_spans:
            pos_ious.append(compute_iou(gt_pos_spans, pred_pos_spans))
        if gt_neg_spans or pred_neg_spans:
            neg_ious.append(compute_iou(gt_neg_spans, pred_neg_spans))
        if gt_all_spans or pred_all_spans:
            if not compute_iou(gt_all_spans, pred_all_spans) == 1:
                print(1)
            all_ious.append(compute_iou(gt_all_spans, pred_all_spans))
        if gt_pos_spans and pred_pos_spans:
            gt_start = sorted(gt_pos_spans, key=lambda x: x[0])[0][0]
            pred_starts = [span[0] for span in pred_pos_spans]
            min_start_diff = min(abs(pred_start - gt_start) for pred_start in pred_starts)
            start_time_diffs.append(min_start_diff)

    # summary
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + (total - tp - fp - fn)) if total else 0.0
    valid_rate = valid / total if total else 0.0
    tn = total - tp - fp - fn
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "准确率 Accuracy": round(accuracy, 4),
        "查准率 Precision": round(precision, 4),
        "查全率 Recall": round(recall, 4),
        "假阳性率 False Positive Rate": round(fpr, 4),
        "可用率 Valid Output Rate": round(valid_rate, 4),
        "异常事件 IoU": round(np.mean(pos_ious), 4) if pos_ious else 1.0,
        "正常事件 IoU": round(np.mean(neg_ious), 4) if neg_ious else 1.0,
        "总体 IoU": round(np.mean(all_ious), 4) if all_ious else 1.0,
        "异常起始偏差秒数": round(np.mean(start_time_diffs), 2) if start_time_diffs else None,
        "总样本数": total
    }

def build_cached_predict_fn(model_path: str) -> Callable[[Dict], str]:
    disable_torch_init()

    from peft import PeftModel
    model, processor = model_init(model_path=model_path)
    model = model.to(torch.bfloat16)
    modal = "video"

    # ✅ 自动生成 cache 文件名
    cache_path = '/home/zhouting/SurveillanceVideoFireAnalysis/VideoLLaMA3/outputs/' + f"cache_{os.path.basename(model_path)}_json.json"

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            pred_cache = json.load(f)
    else:
        pred_cache = {}

    def predict_fn(sample: Dict) -> str:
        sample_id = sample["id"]
        if sample_id in pred_cache:
            return pred_cache[sample_id]

        video_path = os.path.join(
            "/data/zhouting/FireVAD/01_videos_transcoded",
            sample["video"][0]
        )
        frames, timestamps = load_video(video_path, fps=1, max_frames=180)

        # question = "<video>\n请你分析视频片段，判断其中是否发生了与火灾或其他突发情况相关的应急事件。请指出各事件其在视频片段中的时间范围，并简要描述事件内容。若存在多个不同的事件，请按时间顺序输出多个json单元；当视频中发生新的、有意义的、与火灾相关的事件，或当前事件状态发生显著变化时，请开始一个新的事件段。事件段可以重叠。\n每个事件输出要求如下（严格遵循 JSON 格式）：\n{'emergency_exist': '是' 或 '否',  // 是否发生应急事件\n  'event_time': '起始秒-结束秒',  // 时间范围，单位为秒，保留一位小数\n  'event_des': '事件简要描述'  }\n注意事项：时间是相对于该视频片段的局部时间（即片段起点为 0s）；所有事件的范围交集要求覆盖全时段。请仅输出符合上述格式的 JSON。当出现正常、异常事件切换时才区分事件输出，连续相同事件合并为一个事件表述，每个事件表述尽可能短。所有输出不超过70字。"
        question = "<video>\n请你分析视频片段，判断其中是否发生了与火灾或其他突发情况相关的应急事件。请指出各事件其在视频片段中的时间范围，并简要描述事件内容。若存在多个不同的事件，请按时间顺序输出多个json单元；当视频中发生新的、有意义的、与火灾相关的事件，或当前事件状态发生显著变化时，请开始一个新的事件段。事件段可以重叠。\n每个事件输出要求如下（严格遵循 JSON 格式）：\n{'emergency_exist': '是' 或 '否',  // 是否发生应急事件\n  'event_time': '起始秒-结束秒',  // 时间范围，单位为秒，保留一位小数\n  'event_des': '事件简要描述'  }\n注意事项：时间是相对于该视频片段的局部时间（即片段起点为 0s）；所有事件的范围交集要求覆盖全时段。请仅输出符合上述格式的 JSON。当出现正常、异常事件切换时才区分事件输出，连续相同事件合并为一个事件表述，事件表述尽可能短。有烟雾也是异常。"
        # question = "<video>\n请按时间顺序输出多个json单元。\n每个事件输出要求如下（严格遵循 JSON 格式）：\n{'emergency_exist': '是' 或 '否',  // 是否发生应急事件\n  'event_time': '起始秒-结束秒',  // 时间范围，单位为秒，保留一位小数\n  'event_des': '事件简要描述'  }\n注意事项：时间是相对于该视频片段的局部时间（即片段起点为 0s）；所有事件的范围交集要求覆盖全时段。请仅输出符合上述格式的 JSON。如果出现异常烟雾、火光等一定要报告，事件划分尽可能少，每个事件描述要简洁。总长度不超过100字。"
    


        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
                    {"type": "text", "text": question},
                ]
            }
        ]

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
        pred_cache[sample_id] = output.strip()

        # ✅ 自动更新 cache 文件
        with open(cache_path, "w") as f:
            json.dump(pred_cache, f, indent=2, ensure_ascii=False)

        return output.strip()

    return predict_fn

def main():
    dataset_path = "/home/zhouting/SurveillanceVideoFireAnalysis/annotation_data/training_data/StageII/really_training/test/test_samples_1.json"
    
    model_path="/data/zhouting/outputs/fire_analysis/models/videollama3_7b_local_new/directly_ft_stageII_batch32_180f_3e_another_3e_lora"
    lora_adapter_path_stageII = '/data/zhouting/outputs/fire_analysis/models/videollama3_7b_local_new_2stage/stageII_batch32_180f_5e_based_on_stage1_lora'
    base_model_path = "/data/zhouting/models/videollama3_7b_local"
    base_model_path_2b = "/data/zhouting/models/videollama3_2b_local"
    second_2b = '/data/zhouting/outputs/fire_analysis/models/videollama3_2b_local_new_2stage/stageII_batch32_180f_3e_based_on_stage1_2b'
    second_2b = '/data/zhouting/outputs/fire_analysis/models/videollama3_2b_local_new_2stage/stageII_batch32_180f_3e_directly_2b'
    second_2b = '/data/zhouting/outputs/fire_analysis/models/videollama3_2b_local_new_2stage/gptfix_2b_stageII_batch32_180f_3e_directly'
    model_path = second_2b

    with open(dataset_path) as f:
        dataset = json.load(f)

    predict_fn = build_cached_predict_fn(model_path)
    results = evaluate_model_on_dataset(dataset, predict_fn)

    print("\n==== 模型评估结果 ====")
    for k, v in results.items():
        print(f"{k}: {v}")

    # ✅ 保存评估结果到 JSON 文件
    import datetime
    model_id = os.path.basename(model_path)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"/home/zhouting/SurveillanceVideoFireAnalysis/VideoLLaMA3/outputs/eval_results_{model_id}_{now}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 评估结果已保存至: {results_path}")

if __name__ == "__main__":
    main()