import re
import os
import json
from typing import Union, Dict

def extract_think_format(predict_str: str) -> Union[None, Dict[str, str]]:
    """
    检查预测字符串是否符合格式要求，并提取思考部分和回答部分
    
    :param predict_str: 预测的字符串
    :return: 如果符合格式要求，返回包含思考部分和回答部分的字典；否则返回None
    """
    if not predict_str or not isinstance(predict_str, str):
        return None
        
    # 检查<think>是否在开头 
    if not predict_str.startswith("<think>"):
        return None
    
    # 检查是否有 <think>...</think> 格式
    pattern = r"<think>(.*?)</think>"
    think_match = re.search(pattern, predict_str, re.DOTALL)
    if not think_match:
        return None
    
    if predict_str.count("<think>") != 1 or predict_str.count("</think>") != 1:
        return None
    
    # 提取思考部分内容
    think_content = think_match.group(1).strip()
    if not think_content:
        return None
    
    # 获取 </think> 后面的内容
    think_end_pos = predict_str.find("</think>") + len("</think>")
    post_think_content = predict_str[think_end_pos:].strip()
    
    # 检查</think>后面是否有非空内容
    if not post_think_content:
        return None
    
    return {
        "think": think_content,
        "answer": post_think_content
    }

def extract_and_parse_json(input_string, wrapper):
    """
    尝试从字符串中提取并解析 JSON。

    :param input_string: 输入的字符串
    :param wrapper: JSON 的包裹符号，可以是 '{}' 或 '[]'
    :return: 解析后的 JSON 对象，如果解析失败则返回 None
    """
    if len(wrapper) != 2:
        raise ValueError("Wrapper must be exactly two characters long")

    start_char, end_char = wrapper
    start_index = input_string.find(start_char)
    end_index = input_string.rfind(end_char)

    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None

    json_string = input_string[start_index:end_index + 1]

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None

def calculate_iou(bbox1: dict, bbox2: dict) -> float:
    """
    计算两个边界框的交并比(IoU)
    required format: {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    """
    # 计算交集区域
    x_left = max(bbox1["x1"], bbox2["x1"])
    y_top = max(bbox1["y1"], bbox2["y1"])
    x_right = min(bbox1["x2"], bbox2["x2"])
    y_bottom = min(bbox1["y2"], bbox2["y2"])
    
    # 如果没有交集，返回0
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算两个边界框的面积
    bbox1_area = (bbox1["x2"] - bbox1["x1"]) * (bbox1["y2"] - bbox1["y1"])
    bbox2_area = (bbox2["x2"] - bbox2["x1"]) * (bbox2["y2"] - bbox2["y1"])
    
    # 计算并集面积
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou