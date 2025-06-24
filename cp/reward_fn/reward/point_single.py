import logging
import json

from .utils import extract_think_format, extract_and_parse_json

def _extract_verifiable_answer(answer):
    point = extract_and_parse_json(answer, "{}")
    if point is None or "point_2d" not in point:
        return None
    
    point_2d = point["point_2d"]
    if len(point_2d) != 2:
        return None
    
    return point_2d

def _format_reward(answer):
    """
    Calculates the format reward for 'point' type data.

    Args:
        answer (str): The answer string from the model.
    Returns:
        float: The format reward.
    """
    point = _extract_verifiable_answer(answer)
    if point is None:
        return 0.0
    
    return 1.0

def _accuracy_reward(answer, ground_truth):
    """
    Calculates the accuracy reward for 'point' type data.

    Args:
        answer (str): The answer string from the model.
        ground_truth (any): The ground truth for the point.
    Returns:
        float: The accuracy reward.
    """
    pred_point = _extract_verifiable_answer(answer)
    
    if pred_point is None:
        return 0.0, ""
    
    x, y = pred_point
    extracted_answer = json.dumps(pred_point)
    
    if (ground_truth["x1"] <= x <= ground_truth["x2"] and 
        ground_truth["y1"] <= y <= ground_truth["y2"]):
        return 1.0, extracted_answer
    
    return 0.0, extracted_answer

def calculate_reward(solution_str, ground_truth, extra_info=None, fmt_ratio=0.1, acc_ratio=0.9, **kwargs):
    """
    Calculates the reward for 'point' type data.

    Args:
        solution_str (str): The solution string from the model.
        ground_truth (any): The ground truth for the point.
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
