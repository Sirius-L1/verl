import logging
import json
import math
from collections import Counter
from itertools import combinations

from .utils import extract_think_format, extract_and_parse_json

def _extract_verifiable_answer(answer):
    """
    Extracts and verifies the format of the point list from the answer string.
    A valid format is a JSON list of dictionaries, where each dictionary
    has a "point_2d" key with a list of two numbers as the value.
    """
    points = extract_and_parse_json(answer, "[]")
    if points is None or not isinstance(points, list):
        return None
    
    # 验证列表中的每个点
    for point in points:
        if isinstance(point, dict) and "point_2d" in point:
            point_2d = point["point_2d"]
            if isinstance(point_2d, list) and len(point_2d) == 2:
                continue
        
        # If any point is malformed, the whole answer is invalid.
        return None
    
    return points

def _format_reward(answer):
    """
    Calculates the format reward for 'point' type data.
    This function is now primarily used as a check to see if the format is valid.
    Returns 1.0 for a valid format, 0.0 otherwise.
    """
    points = _extract_verifiable_answer(answer)
    if points is None:
        return 0.0
    
    return 1.0

def _check_collinear(points_2d):
    """
    Checks if 3 or more points in the list are collinear on any straight line.
    This uses the cross-product method to avoid division and handle all line types.
    Args:
        points_2d (list): A list of [x, y] coordinates.
    Returns:
        bool: True if 3 or more points are collinear, False otherwise.
    """
    if len(points_2d) < 3:
        return False
    
    # Iterate through all unique combinations of 3 points
    for p1, p2, p3 in combinations(points_2d, 3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Check for collinearity using the cross-product method.
        # If (y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1), the points are collinear.
        # This is equivalent to checking if the area of the triangle formed by the points is 0.
        if (y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1):
            return True
            
    return False

def _accuracy_reward(answer, ground_truth, clip_lower=2, clip_upper=8):
    """
    Calculates the accuracy reward based on the symmetric zero-centered formula.
    The reward is in the range [-1, 1].
    
    Returns a tuple containing:
        - accuracy (float): The calculated reward.
        - extracted_answer (str): The JSON string of the predicted points.
        - num_pred (int): The number of predicted points.
        - first_correct (int): 1 if the first predicted point is correct, 0 otherwise.
        - is_collinear (int): 1 if points are collinear, 0 otherwise.
    """
    pred_points = _extract_verifiable_answer(answer)
    
    # If no valid points are extracted, this is considered a format error, return -1 reward.
    if pred_points is None:
        return -1.0, "", 0, 0, 0

    num_pred = len(pred_points)
    extracted_answer = json.dumps(pred_points)

    if num_pred == 0:
        return -1.0, extracted_answer, 0, 0, 0
    
    # 1. Absolute Penalty: Check for collinearity
    points_2d = [item["point_2d"] for item in pred_points]
    if _check_collinear(points_2d):
        return -1.0, extracted_answer, num_pred, 0, 1

    # 2. Find the rank 'k' of the first correct point
    first_correct_rank = -1
    for i, item in enumerate(pred_points):
        point_2d = item["point_2d"]
        if (ground_truth["x1"] <= point_2d[0] <= ground_truth["x2"] and 
            ground_truth["y1"] <= point_2d[1] <= ground_truth["y2"]):
            first_correct_rank = i + 1  # 1-based index
            break
            
    # 3. Calculate reward based on the zero-centered symmetric formula
    accuracy = 0.0
    if first_correct_rank != -1:
        # Case a: Correct point found (Positive reward space)
        k = first_correct_rank
        accuracy = 1.0 / math.sqrt(num_pred * k)
    else:
        # Case b: No correct point found (Negative reward space)
        accuracy = -1.0 / num_pred
        
    first_correct = 1 if first_correct_rank == 1 else 0
    
    return accuracy, extracted_answer, num_pred, first_correct, 0

def calculate_reward(solution_str, ground_truth, extra_info=None, fmt_ratio=0.0, acc_ratio=1.0, **kwargs):
    """
    Calculates the final reward for 'point' type data.
    Implements the full logic including format checks, collinearity checks,
    and the zero-centered symmetric reward calculation.
    """
    # Set default values for clipping boundaries, can be overridden by kwargs
    clip_lower = kwargs.get('clip_lower', 2)
    clip_upper = kwargs.get('clip_upper', 8)

    if extra_info.get("no_think", False):
        answer = solution_str
    else:
        solution_dict = extract_think_format(solution_str)
        # If the overall 'think'/'answer' format is wrong, return score of -1.
        if solution_dict is None:
            return {
                "score": -1.0, "format": 0.0, "accuracy": -1.0, "pred": "",
                "num_pred": 0, "has_correct": 0, "first_correct": 0,
                "only_correct": 0, "is_collinear": 0
            }
            
        answer = solution_dict["answer"]
    
    # Reuse _format_reward to check the format of the 'answer' part.
    # If it's invalid, return score of -1.
    format_reward = _format_reward(answer)
    if format_reward == 0.0:
        return {
            "score": -1.0, "format": 0.0, "accuracy": -1.0, "pred": "",
            "num_pred": 0, "has_correct": 0, "first_correct": 0,
            "only_correct": 0, "is_collinear": 0
        }
    
    # If format is OK, calculate the accuracy reward.
    accuracy_reward, extracted_answer, num_pred, first_correct, is_collinear = _accuracy_reward(
        answer, ground_truth, clip_lower=clip_lower, clip_upper=clip_upper
    )
    
    return {
        "score": fmt_ratio * format_reward + acc_ratio * accuracy_reward,
        "format": format_reward,
        "accuracy": accuracy_reward,
        "pred": extracted_answer,
        "num_pred": num_pred,
        "has_correct": 1 if accuracy_reward > 0 else 0,
        "first_correct": first_correct,
        "only_correct": 1 if num_pred == 1 and accuracy_reward > 0 else 0,
        "is_collinear": is_collinear
    }
