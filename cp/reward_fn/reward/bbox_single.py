import logging
import json

from .utils import extract_think_format, extract_and_parse_json, calculate_iou

def _extract_verifiable_answer(answer):
    bbox = extract_and_parse_json(answer, "{}")
    if bbox is None or "bbox_2d" not in bbox:
        return None
    
    bbox_2d = bbox["bbox_2d"]
    if len(bbox_2d) != 4:
        return None
    if bbox_2d[0] >= bbox_2d[2] or bbox_2d[1] >= bbox_2d[3]:
        return None
    
    return bbox_2d

def _format_reward(answer):
    """
    Calculates the format reward for 'bbox' type data.

    Args:
        answer (str): The answer string from the model.
    Returns:
        float: The format reward.
    """
    bbox = _extract_verifiable_answer(answer)
    if bbox is None:
        return 0.0
    
    return 1.0

def _accuracy_reward(answer, ground_truth, iou_threshold):
    """
    Calculates the accuracy reward for 'bbox' type data.

    Args:
        answer (str): The answer string from the model.
        ground_truth (any): The ground truth for the bounding box.
    Returns:
        float: The accuracy reward.
    """
    # 提取预测的边界框
    pred_bbox = _extract_verifiable_answer(answer)
    
    if pred_bbox is None:
        return 0.0, ""
    
    # 计算IoU
    bbox = {
        "x1": pred_bbox[0],
        "y1": pred_bbox[1],
        "x2": pred_bbox[2],
        "y2": pred_bbox[3]
    }
    extracted_answer = json.dumps(pred_bbox)
    
    iou = calculate_iou(bbox, ground_truth)
    
    if iou >= iou_threshold:
        return 1.0, extracted_answer
    else:
        return iou / iou_threshold, extracted_answer

def calculate_reward(solution_str, ground_truth, extra_info=None, fmt_ratio=0.1, acc_ratio=0.9, **kwargs):
    """
    Calculates the reward for 'bbox' type data.

    Args:
        solution_str (str): The solution string from the model.
        ground_truth (any): The ground truth for the bounding box.
        extra_info (dict, optional): Additional info.
        fmt_ratio (float, optional): The ratio of format reward.
        acc_ratio (float, optional): The ratio of accuracy reward.
        **kwargs: Additional keyword arguments from config.

    Returns:
        float: The calculated reward.
    """
    solution_dict = extract_think_format(solution_str)
    if solution_dict is None:
        return {
            "score": 0.0,
            "format": 0.0,
            "accuracy": 0.0,
            "pred": ""
        }
    thinking = solution_dict["think"]
    answer = solution_dict["answer"]
    
    format_reward = _format_reward(answer)
    if format_reward == 0.0:
        return {
            "score": 0.0,
            "format": 0.0,
            "accuracy": 0.0,
            "pred": ""
        }
    
    accuracy_reward, extracted_answer = _accuracy_reward(answer, ground_truth, iou_threshold=kwargs.get("iou_threshold", 0.7))
    
    return {
        "score": fmt_ratio * format_reward + acc_ratio * accuracy_reward,
        "format": format_reward,
        "accuracy": accuracy_reward,
        "pred": extracted_answer
    }
