import json
import os
from typing import Dict, Any, List, Optional

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {file_path}")
        return {}

def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """Save data to a JSON file.
    
    Args:
        data: Dictionary to save as JSON
        file_path: Path where to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False

def format_search_results(search_results: List[Dict[str, Any]]) -> str:
    """Format search results into a readable string.
    
    Args:
        search_results: List of search result dictionaries
        
    Returns:
        Formatted string with search results
    """
    formatted_text = "검색 결과:\n\n"
    
    for i, result in enumerate(search_results):
        formatted_text += f"[결과 {i+1}] "
        
        if "title" in result and result["title"]:
            formatted_text += f"제목: {result['title']}"
            if "page_num" in result and result["page_num"]:
                formatted_text += f" (페이지: {result['page_num']})"
            formatted_text += "\n"
        
        if "text" in result and result["text"]:
            # Trim text if too long
            text = result["text"]
            if len(text) > 300:
                text = text[:300] + "..."
            formatted_text += f"내용: {text}\n"
        
        if "similarity" in result:
            formatted_text += f"유사도: {result['similarity']:.4f}\n"
        
        formatted_text += "\n"
    
    return formatted_text
