import logging

from .utils import extract_think_format, extract_and_parse_json

def _format_reward(answer):
    """
    Calculates the format reward for 'point' type data.

    Args:
        answer (str): The answer string from the model.
    Returns:
        float: The format reward.
    """
    point = extract_and_parse_json(answer, "[]")
    if point is None or len(point) != 1:
        return 0.0
    
    for point_item in point:
        if "point_2d" not in point_item:
            return 0.0
            
        point_2d = point_item["point_2d"]
        if len(point_2d) != 2:
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
    pred_point = extract_and_parse_json(answer, "[]")
    
    if pred_point is None or len(pred_point) != 1:
        return 0.0
    
    # 计算每个预测点是否在框内
    points_in_bbox = 0
    for point_item in pred_point:
        if "point_2d" not in point_item:
            continue
            
        point = point_item["point_2d"]
        if len(point) != 2:
            continue
            
        x, y = point
        
        # 检查点是否在边界框内
        if (ground_truth["x1"] <= x <= ground_truth["x2"] and 
            ground_truth["y1"] <= y <= ground_truth["y2"]):
            points_in_bbox += 1
    
    try:
        extracted_answer = pred_point[0]["point_2d"]
    except Exception as e:
        extracted_answer = None

    # 计算得分
    if points_in_bbox > 0:
        return 1.0, extracted_answer
    else:
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
            "pred": None
        }
    thinking = solution_dict["think"]
    answer = solution_dict["answer"]
    
    format_reward = _format_reward(answer)
    if format_reward == 0.0:
        return {
            "score": 0.0,
            "format": 0.0,
            "accuracy": 0.0,
            "pred": None
        }
    
    accuracy_reward, extracted_answer = _accuracy_reward(answer, ground_truth)
    
    return {
        "score": fmt_ratio * format_reward + acc_ratio * accuracy_reward,
        "format": format_reward,
        "accuracy": accuracy_reward,
        "pred": extracted_answer
    }
