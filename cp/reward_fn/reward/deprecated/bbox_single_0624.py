import logging
import json

from .utils import extract_think_format, extract_and_parse_json, calculate_iou

def _format_reward(answer):
    """
    Calculates the format reward for 'bbox' type data.

    Args:
        answer (str): The answer string from the model.
    Returns:
        float: The format reward.
    """
    bbox = extract_and_parse_json(answer, "[]")
    if bbox is None or len(bbox) != 1:
        return 0.0
    
    # 检查边界框的有效性：x1 < x2 且 y1 < y2
    for bbox_item in bbox:
        if "bbox_2d" not in bbox_item:
            return 0.0
            
        bbox_2d = bbox_item["bbox_2d"]
        if len(bbox_2d) != 4:
            return 0.0
            
        if bbox_2d[0] >= bbox_2d[2] or bbox_2d[1] >= bbox_2d[3]:
            return 0.0
    
    return 1.0

def _accuracy_reward(answer, ground_truth, iou_threshold=0.7):
    """
    Calculates the accuracy reward for 'bbox' type data.

    Args:
        answer (str): The answer string from the model.
        ground_truth (any): The ground truth for the bounding box.
    Returns:
        float: The accuracy reward.
    """
    # 提取预测的边界框
    pred_bbox = extract_and_parse_json(answer, "[]")
    
    if pred_bbox is None or len(pred_bbox) != 1:
        return 0.0
    
    # 计算IoU
    max_iou = 0.0
    for bbox_item in pred_bbox:
        pred_bbox_2d = bbox_item["bbox_2d"]
        pred_box = {
            "x1": pred_bbox_2d[0],
            "y1": pred_bbox_2d[1],
            "x2": pred_bbox_2d[2],
            "y2": pred_bbox_2d[3]
        }
        
        iou = calculate_iou(pred_box, ground_truth)
        max_iou = max(max_iou, iou)
    
    try:
        extracted_answer = json.dumps(pred_bbox[0]["bbox_2d"])
    except Exception as e:
        extracted_answer = ""
    
    # 根据IoU计算得分
    if max_iou >= iou_threshold:
        return 1.0, extracted_answer
    else:
        return max_iou / iou_threshold, extracted_answer

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
    
    accuracy_reward, extracted_answer = _accuracy_reward(answer, ground_truth)
    
    return {
        "score": fmt_ratio * format_reward + acc_ratio * accuracy_reward,
        "format": format_reward,
        "accuracy": accuracy_reward,
        "pred": extracted_answer
    }
