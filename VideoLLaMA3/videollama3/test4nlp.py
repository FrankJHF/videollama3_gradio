import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


def contains_positive_kw(text: str, keywords: list, negation_words=["未", "不", "没有", "未发现", "无"]):
    """
    判断文本中是否存在没有被否定的关键词（如“存在异常”前没有“不”之类否定词）
    """
    for kw in keywords:
        for match in re.finditer(re.escape(kw), text):
            start_idx = match.start()
            preceding_text = text[max(0, start_idx - 5):start_idx]  # 取前面最多5个字符
            if not any(neg in preceding_text for neg in negation_words):
                return True
    return False

def extract_event_spans_from_text(text: str, max_gt_time):
    """
    从模型输出的自由文本中提取异常事件 span 列表
    返回两个列表：异常事件 spans，正常事件 spans（通常只分段全片）
    """
    # 判断是否有异常
    emergency_exist = contains_positive_kw(text, ["存在异常", "发现异常","是，","是,","存在"])
    
    # 提取所有类似“第xx秒到yy秒”、“xxs-yy秒”、“xx到yy秒”格式的区间
    time_spans = []
    pattern = re.findall(
    r"第?\s*(\d+(?:\.\d+)?)\s*(?:秒|s)?\s*(?:到|至|-|~)\s*第?\s*(\d+(?:\.\d+)?)\s*(?:秒|s)?",
    text)
    
    for match in pattern:
        try:
            start, end = float(match[0]), float(match[1])
            if start > end:
                start, end = end, start
            time_spans.append([start, end])
        except:
            continue

    if emergency_exist:
        return time_spans, []
    else:
        return [], [[0.0, max_gt_time]]  # 没有异常时认为全段是正常
    

import json
import numpy as np
from typing import List, Dict, Callable
from tqdm import tqdm

def evaluate_model_on_dataset(
    dataset: List[Dict],
    predict_fn: Callable[[Dict], str]
) -> (Dict[str, float], List[Dict]):
    total = len(dataset)
    tp = fp = fn = valid = 0
    pos_ious = []
    neg_ious = []
    all_ious = []
    start_time_diffs = []  # 记录所有有对齐异常的起始时刻差

    per_sample_details = []

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
        gt_pos_spans = [s for s in gt_pos_spans if s]
        gt_neg_spans = [s for s in gt_neg_spans if s]
        gt_all_spans = gt_pos_spans + gt_neg_spans
        max_gt_time = max((s[1] for s in gt_all_spans), default=180.0)

        gt_emergency = len(gt_pos_spans) > 0

        try:
            pred_str = predict_fn(item)
            print(pred_str)
            valid += 1
        except Exception:
            pred_str = ""

        # 从自然语言输出中提取时间段
        pred_pos_spans, pred_neg_spans = extract_event_spans_from_text(pred_str, max_gt_time)
        pred_pos_spans = [span for span in pred_pos_spans if span]
        pred_neg_spans = [span for span in pred_neg_spans if span]
        pred_all_spans = pred_pos_spans + pred_neg_spans

        pred_emergency = len(pred_pos_spans) > 0
        cls_correct = (gt_emergency == pred_emergency)

        # classification-level指标
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


        # pos_iou = compute_iou(gt_pos_spans, pred_pos_spans) if (gt_pos_spans or pred_pos_spans) else 1.0
        # neg_iou = compute_iou(gt_neg_spans, pred_neg_spans) if (gt_neg_spans or pred_neg_spans) else 1.0
        # all_iou = compute_iou(gt_all_spans, pred_all_spans) if (gt_all_spans or pred_all_spans) else 1.0
        # pos_ious.append(pos_iou)
        # neg_ious.append(neg_iou)
        # all_ious.append(all_iou)

        # 格式化 span 为字符串
        def fmt_spans(spans): return ", ".join([f"{s[0]}s-{s[1]}s" for s in spans])

        per_sample_details.append({
            "样本ID": item["id"],
            "GT是否异常": gt_emergency,
            "预测是否异常": pred_emergency,
            "分类是否正确": cls_correct,
            "GT异常段": fmt_spans(gt_pos_spans),
            "预测异常段": fmt_spans(pred_pos_spans),
            "GT正常段": fmt_spans(gt_neg_spans),
            "预测正常段": fmt_spans(pred_neg_spans),
            "异常事件 IoU": round(np.mean(pos_ious), 4) if pos_ious else 1.0,
            "正常事件 IoU": round(np.mean(neg_ious), 4) if neg_ious else 1.0,
            "总体 IoU": round(np.mean(all_ious), 4) if all_ious else 1.0,
            "模型原始输出": pred_str.strip().replace("\n", "")
        })

        # print(per_sample_details[-1])

    # summary
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + (total - tp - fp - fn)) if total else 0.0
    valid_rate = valid / total if total else 0.0

    tn = total - tp - fp - fn
    accuracy = (tp + tn) / total if total else 0.0

    metrics = {
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

    return metrics, per_sample_details

def build_cached_predict_fn(model_path: str) -> Callable[[Dict], str]:
    disable_torch_init()

    from peft import PeftModel
    model, processor = model_init(model_path=model_path)
    model = model.to(torch.bfloat16)
    modal = "video"

    # ✅ 自动生成 cache 文件名
    cache_path = '/home/zhouting/SurveillanceVideoFireAnalysis/VideoLLaMA3/outputs/' + f"cache_{os.path.basename(model_path)}_nlp.json"

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
        # p1-question = "<video>\n请你分析视频片段，判断其中是否发生了与火灾或其他突发情况相关的应急事件。请指出各事件其在视频片段中的时间范围，并简要描述事件内容。若存在多个不同的事件，请按时间顺序输出多个json单元；当视频中发生新的、有意义的、与火灾相关的事件，或当前事件状态发生显著变化时，请开始一个新的事件段。事件段可以重叠。\n每个事件输出要求如下（严格遵循 JSON 格式）：\n{'emergency_exist': '是' 或 '否',  // 是否发生应急事件\n  'event_time': '起始秒-结束秒',  // 时间范围，单位为秒，保留一位小数\n  'event_des': '事件简要描述'  }\n注意事项：时间是相对于该视频片段的局部时间（即片段起点为 0s）；所有事件的范围交集要求覆盖全时段。请仅输出符合上述格式的 JSON。当出现正常、异常事件切换时才区分事件输出，连续相同事件合并为一个事件表述，事件表述尽可能短。有烟雾也是异常。"
        question = "<video>\n请按时间顺序输出多个json单元。\n每个事件输出要求如下（严格遵循 JSON 格式）：\n{'emergency_exist': '是' 或 '否',  // 是否发生应急事件\n  'event_time': '起始秒-结束秒',  // 时间范围，单位为秒，保留一位小数\n  'event_des': '事件简要描述'  }\n注意事项：时间是相对于该视频片段的局部时间（即片段起点为 0s）；所有事件的范围交集要求覆盖全时段。请仅输出符合上述格式的 JSON。如果出现异常烟雾、火光等一定要报告，总长度不超过100字。"
        question = "视频中是否存在异常？有无与火灾相关事件？在视频的第几秒到第几秒时候发生？回答尽量短。"


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
    base_model_path_2b = "/data/zhouting/models/videollama3_2b_local"
    base_model_path_7b = "/data/zhouting/models/videollama3_7b_local"

    second_2b = '/data/zhouting/outputs/fire_analysis/models/videollama3_2b_local_new_2stage/stageII_batch32_180f_3e_based_on_stage1_2b'
    model_path = base_model_path_7b

    with open(dataset_path) as f:
        dataset = json.load(f)

    predict_fn = build_cached_predict_fn(model_path)
    results, details = evaluate_model_on_dataset(dataset, predict_fn)

    print("\n==== 模型评估结果 ====")
    for k, v in results.items():
        print(f"{k}: {v}")

    # ✅ 保存评估结果到 JSON 文件
    import datetime
    model_id = os.path.basename(model_path)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"/home/zhouting/SurveillanceVideoFireAnalysis/VideoLLaMA3/outputs/eval_results_{model_id}_nlp_{now}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 评估结果已保存至: {results_path}")

    # 保存样本详情
    details_path = f"/home/zhouting/SurveillanceVideoFireAnalysis/VideoLLaMA3/outputs/eval_details_{model_id}_{now}.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()