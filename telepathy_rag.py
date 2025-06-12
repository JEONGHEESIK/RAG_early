import os, json, sys, re, time, datetime, requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel
import argparse
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
import faiss
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from accelerate import infer_auto_device_map

# Import all necessary modules from the numbered files
# Import the modules with their full path since they have numbers in their names
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules using importlib to handle module names with numbers
import importlib.machinery

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import each module using absolute paths
rag_config = importlib.machinery.SourceFileLoader(
    'rag_config', 
    os.path.join(current_dir, '1_rag_config.py')
).load_module()

rag_search = importlib.machinery.SourceFileLoader(
    'rag_search', 
    os.path.join(current_dir, '2_rag_search.py')
).load_module()

rag_formatter = importlib.machinery.SourceFileLoader(
    'rag_formatter', 
    os.path.join(current_dir, '3_rag_formatter.py')
).load_module()

rag_rewriter = importlib.machinery.SourceFileLoader(
    'rag_rewriter', 
    os.path.join(current_dir, '4_rag_rewriter.py')
).load_module()

rag_fastapi = importlib.machinery.SourceFileLoader(
    'rag_fastapi', 
    os.path.join(current_dir, '5_rag_fastapi.py')
).load_module()

rag_initializer = importlib.machinery.SourceFileLoader(
    'rag_initializer', 
    os.path.join(current_dir, '6_rag_initializer.py')
).load_module()

rag_integration = importlib.machinery.SourceFileLoader(
    'rag_integration', 
    os.path.join(current_dir, '7_rag_integration.py')
).load_module()

rag_clihandler = importlib.machinery.SourceFileLoader(
    'rag_clihandler', 
    os.path.join(current_dir, '8_rag_clihandler.py')
).load_module()

ollamagen = importlib.machinery.SourceFileLoader(
    'ollamagen', 
    os.path.join(current_dir, '9_ollamagen.py')
).load_module()

image_incoder = importlib.machinery.SourceFileLoader(
    'image_incoder', 
    os.path.join(current_dir, '10_image_incoder.py')
).load_module()

# Import the necessary classes and functions
from rag_config import RAGConfig
from rag_search import RAGSearchEngine
from rag_formatter import format_search_results
from rag_rewriter import rewrite_query
from rag_fastapi import app, SearchRequest, SearchResponse, RAGRequest, RAGResponse
from rag_initializer import config, search_engine
from rag_integration import SearchIntegration
from rag_clihandler import CLIHandler, parse_arguments, cli_search, interactive_mode, list_metadata
from ollamagen import OllamaGenerator
from image_incoder import (
    resize_image,
    encode_image_to_base64,
    extract_image_paths,
    process_response_with_images
)

def main():
    """Main entry point for the RAG system."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize the search integration
    search_integration = SearchIntegration(search_engine)
    
    # Initialize the CLI handler
    cli_handler = CLIHandler(config, search_engine, search_integration)
    
    # Handle command line arguments
    if args.list_metadata:
        # List metadata and exit
        list_metadata()
        return
    
    if args.interactive:
        # Start interactive mode
        interactive_mode()
    elif args.query:
        # Process a single query
        cli_search(
            query=args.query,
            top_k=args.top_k,
            show_images=not args.no_images,
            use_exact_match=args.exact_match,
            media_type=args.media_type,
            generate_answer=not args.no_ai
        )
    else:
        print("No query provided. Use --help for usage information.")

if __name__ == "__main__":
    main()

# For FastAPI integration
app.include_router(router, prefix="/api")

# For backward compatibility
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
