import argparse
import os
import re

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils import write_tensor_to_file


def dump_inputs(data_dir, output_dir, num_samples):
    # Walk the category directories and collect the image paths
    image_paths = []
    for dir, _, files in os.walk(data_dir):
        for f in files:
            image_paths.append(os.path.join(dir, f))
    
    # Only make single sample case deterministic (as per Kartik's request)
    if num_samples == 1:
        image_paths = sorted(image_paths)

    # Get (all) reference labels from the labels file in the root of the data directory
    ref_labels = []
    with open(image_paths[0], "r") as f:
        ref_labels = [s.strip().split(',') for s in f.readlines()]

    assert len(image_paths) >= num_samples, (
        f'Can not generate {num_samples}, only {len(image_paths)} samples available.'
    )

    # Remove label file from path list
    image_paths = image_paths[1:]

    # Get label to every image by extracting it from the path
    image_labels = [re.findall('n[0-9]+', i)[0] for i in image_paths]

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
        class_labels = f.readlines()
        class_labels = [s.split(',')[0] for s in class_labels]

    for i in tqdm(range(num_samples)):
        # Load and prepare input image
        image_path = image_paths[i]
        input_image = Image.open(image_path).convert('RGB')
        input = preprocess(input_image)

        # Get unique name from image path
        path_components = os.path.normpath(image_path).split(os.path.sep)
        model_id = path_components[-2] + '_' + re.findall('000[0-9]+', path_components[-1])[-1]

        # Write inputs to file
        class_id = class_labels.index(image_labels[i])
        dir_name = os.path.join(output_dir, f"{class_id}_{model_id}")
        os.makedirs(dir_name, exist_ok=True)
        write_tensor_to_file(input, os.path.join(dir_name, "arg0_1.bin"))


def main():
    parser = argparse.ArgumentParser(
        description='ResNet data generator (from model and images to binary data).'
    )
    parser.add_argument(
        '--data_dir',
        default='data/imagenet',
        help='Path to dataset.',
    )
    parser.add_argument(
        '--output_dir',
        default='data/imagenet',
        help='Path to output folder for dataset.',
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1,
        help='How many samples to generate.',
    )
    args = parser.parse_args()
    dump_inputs(args.data_dir, args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()
