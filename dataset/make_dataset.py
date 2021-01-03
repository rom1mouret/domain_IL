#!/usr/bin/env python3

import sys
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageOps
from collections import defaultdict

directory = sys.argv[1]
train_dir = os.path.join(directory, "Training")
test_dir = os.path.join(directory, "Test")

hand = Image.open("hand.png")
frame = Image.open("painting_frame2.jpeg")
cache = {}

def resize_hand(w: int, h: int):
    key = ("hand", w, h)
    resized_hand = cache.get(key)
    if resized_hand is None:
        resized_hand = hand.resize((w, h))
        cache[key] = resized_hand
    return resized_hand

def resize_frame(w: int, h: int):
    key = ("frame", w, h)
    resized_frame = cache.get(key)
    if resized_frame is None:
        resized_frame = frame.resize((w, h))
        cache[key] = resized_frame
    return resized_frame

def add_margin(pil_img, margin: int):
    width, height = pil_img.size
    new_width = width + 2 * margin
    new_height = height + 2 * margin
    result = Image.new(pil_img.mode, (new_width, new_height), (255,255,255))
    result.paste(pil_img, (margin, margin))
    return result

# image paths
classes_to_images = defaultdict(list)
for dir in (train_dir, test_dir):
    for class_dir in os.listdir(dir):
        for image_file in os.listdir(os.path.join(dir, class_dir)):
            path = os.path.join(dir, class_dir, image_file)
            classes_to_images[class_dir].append(path)


# processing/writing function
def generate_images(
        outdir: str,
        classes: list,
        sample_filtering=lambda li: li,
        market_selector=lambda n: n % 2 == 0,
        museum_selector=lambda n: n % 2 == 1):
    max_length = 64
    margin = 50
    for klass in tqdm(classes, total=len(classes)):
        for n, image_path in enumerate(sample_filtering(classes_to_images[klass])):
            image_file = os.path.basename(image_path)
            image = Image.open(image_path)
            width, height = image.size
            if max(width, height) > max_length:
                if height > width:
                    ratio = height / max_length
                else:
                    ratio = width / max_length
                width = int(width / ratio)
                height = int(height / ratio)
                image = image.resize((width, height))
            image = add_margin(image, margin)
            width += 2 * margin
            height += 2 * margin

            gray = image.convert("L")
            npmask = (np.asarray(gray) < 250).astype(np.uint8) * 255
            mask = Image.fromarray(npmask)
            if market_selector(n):
                marketdir = os.path.join(outdir, "market", klass)
                os.makedirs(marketdir, exist_ok=True)
                resized_hand = resize_hand(width, height)
                market = Image.composite(image, resized_hand, mask)
                market.convert('RGB').save(os.path.join(marketdir, image_file))
            if museum_selector(n):
                museumdir = os.path.join(outdir, "museum", klass)
                os.makedirs(museumdir, exist_ok=True)
                resized_frame = resize_frame(width, height)
                edges = ImageOps.solarize(image, threshold=90)
                edges = edges.filter(ImageFilter.BLUR)
                museum = Image.composite(edges, resized_frame, mask)
                museum.convert('RGB').save(os.path.join(museumdir, image_file))


# split the dataset into merging experiment and zero-shot learning experiment
classes = sorted(classes_to_images.keys())
random.seed(0)
random.shuffle(classes)
cut = len(classes) // 10
merging_classes, zeroshot_classes = classes[cut:], classes[:cut]

# zero-shot learning experiment
generate_images("zeroshot", zeroshot_classes)

# merging experiment
def training_selector(li: list) -> list:
    return li[:-len(li) // 10]

def testing_selector(li: list) -> list:
    for_training = set(training_selector(li))
    return [path for path in li if path not in for_training]

def select_all(n: int) -> bool:
    return True

generate_images("training", merging_classes, sample_filtering=training_selector)
generate_images("testing", merging_classes, sample_filtering=testing_selector,
                market_selector=select_all, museum_selector=select_all)
