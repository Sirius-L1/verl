import logging
import json

from .utils import extract_think_format, extract_and_parse_json

def _extract_verifiable_answer(answer):
    points = extract_and_parse_json(answer, "[]")
    if points is None or not isinstance(points, list) or len(points) == 0:
        return None
    
    # 验证列表中的每个点
    for point in points:
        if isinstance(point, dict) and "point_2d" in point:
            point_2d = point["point_2d"]
            if isinstance(point_2d, list) and len(point_2d) == 2:
                continue
        
        return None
    
    return points

def _format_reward(answer):
    """
    Calculates the format reward for 'point' type data.

    Args:
        answer (str): The answer string from the model.
    Returns:
        float: The format reward.
    """
    points = _extract_verifiable_answer(answer)
    if points is None:
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
        str: The extracted answer as JSON string.
    """
    pred_points = _extract_verifiable_answer(answer)
    
    if pred_points is None:
        return 0.0, ""
    
    extracted_answer = json.dumps(pred_points)
    
    # 检查是否有任何预测点在ground_truth范围内
    has_correct_point = False
    for x, y in pred_points:
        if (ground_truth["x1"] <= x <= ground_truth["x2"] and 
            ground_truth["y1"] <= y <= ground_truth["y2"]):
            has_correct_point = True
            break
    
    if has_correct_point:
        # 如果有正确的点，分数为 1/n，其中n为总预测点数量
        return 1.0 / len(pred_points), extracted_answer, len(pred_points)
    else:
        # 如果没有正确的点，分数为0
        return 0.0, extracted_answer, len(pred_points)

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
            "pred": "",
            "num_pred": 0,
            "has_correct": 0
        }
    thinking = solution_dict["think"]
    answer = solution_dict["answer"]
    
    format_reward = _format_reward(answer)
    if format_reward == 0.0:
        return {
            "score": 0.0,
            "format": 0.0,
            "accuracy": 0.0,
            "pred": "",
            "num_pred": 0,
            "has_correct": 0
        }
    
    accuracy_reward, extracted_answer, num_pred = _accuracy_reward(answer, ground_truth)
    
    return {
        "score": fmt_ratio * format_reward + acc_ratio * accuracy_reward,
        "format": format_reward,
        "accuracy": accuracy_reward,
        "pred": extracted_answer,
        "num_pred": num_pred,
        "has_correct": 1 if accuracy_reward > 0 else 0
    }
