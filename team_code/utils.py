import os
import sys
import time
import json
import torch
import argparse
import numpy as np
import traceback as tr
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Union, Iterable, Optional, Dict, Any
from huggingface_hub.utils import HFValidationError

sys.path.append(str(Path(__file__)))
os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.getcwd())


def process_data_sample(sample: Dict[str, Any],
                        processor,
                        modality_type: Optional[str] = 'video',
                        **kwargs) -> Dict[str, Any]:
    """
    Preprocessing function for visual/audio modalities and text instruction.
    :param sample: a data sample;
    Example structure for multiple-choice QA:
        {
            'task_id': 1,
            'task_type': 'qa',
            'question': 'Who is dancing in the end of the video?',
            'video': 'path_to_video.mp4',
            'audio': 'path_to_audio.mp3',
            'choices': [
               {'choice_id': 1, 'choice': 'Woman in red'},
               {'choice_id': 2, 'choice': 'Woman in blue'},
               {'choice_id': 3, 'choice': 'Man in green'},
               {'choice_id': 4, 'choice': 'Man in black'},
               {'choice_id': 5, 'choice': 'Nobody'},
            ] ,
            'correct_answer': 'Woman in blue',
           'correct_answer_choice_id': 2
         }

         or for captioning

        {
          'task_id': 2,
          'task_type': 'captioning',
          'question': 'Describe this video in detail.',
          'video': 'path_to_video.mp4',
          'audio': '',
          'choices': []
        }
    :param processor: processor as a dict of multiple processing functions;
    :param modality_type: a type of requested modality, one from ['image', 'video', 'audio'];
    :param kwargs: addition arguments;
    :return: processed_sample as a Dict[str, Any].
    """
    # 1. Get modality and process it
    modality_path = sample.get(modality_type, None)
    if modality_path is None:
        raise ValueError(f"Modality type: `{modality_type}` was not found in input data sample!")

    if not os.path.exists(modality_path):
        print(f"Error while loading modality (`{modality_type}`) data, can't find a file by path: `{modality_path}`")
        raise FileNotFoundError(f"Error while loading modality (`{modality_type}`) data, can't find a file by path: `{modality_path}`")

    modality_features = processor[modality_type](modality_path)
    if 'return_dtype' in kwargs:
        try:
            return_dtype = getattr(torch, kwargs['return_dtype'])
            modality_features = video_tensor.to(dtype=return_dtype)
        except AttributeError as e:
            print(f"Invalid data type passed: `{kwargs['return_dtype']}`")

    # 2. Construct instruction for task
    if sample['task_type'] == 'qa':
        # Run construction of multiple-choice QA query
        question = sample['question']
        choices = [choice['choice'] for choice in sample['choices']]
        options = ['(A)', '(B)', '(C)', '(D)', '(E)']
        instruction = f"Question: {question}\nOptions:"
        for opt, ans in zip(options, choices):
            instruction += f"\n{opt} {ans}"
        instruction += "\nAnswer with the option\'s letter from the given choices directly and only give the best option."
        return {
            'task_id': sample['task_id'],
            'task_type': sample['task_type'],
            modality_type: modality_features,
            'instruction': instruction,
            'answers': [f"{o} {a}" for o, a in zip(options, choices)]
        }
    elif sample['task_type'] == 'captioning':
        # Run construction of captioning query
        question = sample['question']
        instruction = f"Question: {question}\nAnswer: "
        return {
            'task_id': sample['task_id'],
            'task_type': sample['task_type'],
            modality_type: modality_features,
            'instruction': instruction
        }
