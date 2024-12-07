import re
import json
import random
import multiprocessing
import numpy as np
from tqdm import tqdm
from pyvi import ViTokenizer
from collections import Counter


BATCH_SIZE = 128
CONTEXT_WINDOW_SIZE = 2
VOCAB_SIZE = 50000

input_path = '../data/news_dataset_lite.json'
output_data = '../data/processed_data_lite.json'
output_vocab = '../data/vocab_lite.json'


def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # Loại bỏ URL
    text = re.sub(r"\([^)]*\)", " ", text)  # Loại bỏ nội dung trong ngoặc
    text = re.sub(r"[^\w\s]", " ", text)  # Loại bỏ dấu câu
    text = re.sub(r"\d+", " ", text)  # Loại bỏ số
    return text.strip()

def process_item(item):
    item['content'] = ViTokenizer.tokenize(item['content'])
    item['title'] = ViTokenizer.tokenize(item['title'])
    return item

def parallel_processing(data):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_item, data), total=len(data)))
    return results

def generate_sample(index_vocab, context_window_size):
    for index, center in enumerate(index_vocab):
        context = random.randint(1, context_window_size)
        for target in index_vocab[max(0, index - context): index]:
            yield center, target
        for target in index_vocab[index + 1: index + context + 1]:
            yield center, target

def get_batch(iterator, batch_size):
    while True:
        center_batch = np.zeros([batch_size], dtype=np.int32)
        target_batch = np.zeros([batch_size, 1], dtype=np.int32)
        for ind in range(batch_size):
            center_batch[ind], target_batch[ind] = next(iterator)
        yield center_batch, target_batch

