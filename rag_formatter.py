import os, json, sys, re, time, datetime, requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
import requests
import argparse
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import faiss
from fastapi import FastAPI, Request


# 6. 검색 결과 포맷팅 유틸리티 함수
def format_search_results(search_results: List[Dict[str, Any]]) -> str:
    """검색 결과를 읽기 쉬운 형식으로 포맷팅"""
    formatted_text = "검색 결과:\n\n"
    
    for i, result in enumerate(search_results):
        formatted_text += f"[결과 {i+1}] "
        
        if "title" in result and result["title"]:
            formatted_text += f"제목: {result['title']}"
            if "page_num" in result and result["page_num"]:
                formatted_text += f" (페이지: {result['page_num']})"
            formatted_text += "\n"
        
        if "text" in result and result["text"]:
            # 텍스트가 너무 길면 잘라서 표시
            text = result["text"]
            if len(text) > 300:
                text = text[:300] + "..."
            formatted_text += f"내용: {text}\n"
        
        if "similarity" in result:
            formatted_text += f"유사도: {result['similarity']:.4f}\n"
        
        formatted_text += "\n"
    
    return formatted_text