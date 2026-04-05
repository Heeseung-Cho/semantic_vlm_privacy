import os
import json
import re
import numpy as np
import cv2


FIELD_MARKERS = (
    " - position:",
    " - confidence:",
    " - features:",
    " position:",
    " confidence:",
    " features:",
)


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def parse_response(response_text):
    # Use regex to extract content between <output> and </output>
    match = re.search(r'<output>(.*?)</output>', response_text, re.DOTALL)
    
    if match:
        raw_output = match.group(1).strip()
        lowered_output = raw_output.lower()
        if lowered_output == "no objects matching the given categories could be identified":
            return raw_output

        normalized_output = raw_output.replace("\n", ",")
        
        items = []
        for item in normalized_output.split(','):
            cleaned = item.strip().strip('[]')
            cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
            lowered = cleaned.lower()
            for marker in FIELD_MARKERS:
                marker_idx = lowered.find(marker)
                if marker_idx != -1:
                    cleaned = cleaned[:marker_idx].strip()
                    lowered = cleaned.lower()
                    break
            if cleaned:
                items.append(cleaned)

        deduped_items = []
        seen = set()
        for item in items:
            key = item.lower()
            if key not in seen:
                deduped_items.append(item)
                seen.add(key)
        return ', '.join(deduped_items) if deduped_items else "No output found"
    else:
        return "No output found"

def mask_to_coco_polygon(args,mask):
    """
    Convert a binary mask to COCO format polygons.
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        if "realesrgan" in args.query_dir:
            contour = contour / args.scale
        else:
            contour = contour
        polygon = contour.flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons

def create_gpt_log_summary(processed_files, output_dir, log_dir=None):

    if log_dir is None:
        log_dir = os.path.join(output_dir, "gpt_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # use json file name as summary file name
    summary_file_path = os.path.join(log_dir, "gpt_summary.json")
    sum={}
    for idx, (file_name, text_prompt) in enumerate(processed_files, 1):
        sum[file_name] = text_prompt
    with open(summary_file_path, 'w', encoding='utf-8') as f:   
        json.dump(sum, f, ensure_ascii=False, indent=2)
    
    print(f"GPT processing summary saved to {summary_file_path}")

def log_gpt_output(image_path, response_text, parsed_text, output_dir, log_dir=None):

    if log_dir is None:
        log_dir = os.path.join(output_dir, "gpt_logs")
    os.makedirs(log_dir, exist_ok=True)
    

    file_name = os.path.basename(image_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    

    log_file_path = os.path.join(log_dir, f"{file_name_no_ext}_gpt_output.txt")
    

    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Image: {file_name}\n")
        f.write(f"Full GPT Response:\n")
        f.write("-" * 80 + "\n")
        f.write(response_text)
        f.write("\n" + "-" * 80 + "\n")
        f.write(f"Parsed Categories: {parsed_text}\n")
        f.write("-" * 80 + "\n")
    
    print(f"GPT output for {file_name} logged to {log_file_path}")
