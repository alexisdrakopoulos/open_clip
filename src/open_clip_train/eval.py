import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import logging
from PIL import Image
from tqdm import tqdm
import faiss
from torch.utils.data import Dataset, DataLoader
from open_clip_train.distributed import is_master


logger = logging.getLogger(__name__)

# --- Helper Classes and Functions (adapted from your script) ---


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


class ImageDataset(Dataset):
    """A simple dataset to load images from a list of paths."""

    def __init__(self, image_paths, transform, root_path=""):
        self.image_paths = image_paths
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Allow for absolute or relative paths
        path = os.path.join(self.root_path, self.image_paths[idx])
        try:
            image = Image.open(path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            logger.warning(
                f"Could not load image {path}, returning a blank tensor. Error: {e}"
            )
            # Return a dummy tensor with the correct dimensions
            # We can infer the size from the transform's CenterCrop
            size = 224  # default
            for t in self.transform.transforms:
                if hasattr(t, "size"):
                    # Handle int or tuple size
                    size = t.size if isinstance(t.size, int) else t.size[0]
            return torch.zeros((3, size, size))


def calculate_recall_at_k(retrieved_items, ground_truth_items, k):
    """Calculates if any of the top-k retrieved items are in the ground truth set."""
    return len(set(retrieved_items[:k]) & set(ground_truth_items)) > 0


# --- Main Evaluation Function for OpenCLIP ---


def run_image_retrieval_evaluation(model, transform, epoch, args):
    """
    Runs a full image retrieval evaluation benchmark for an OpenCLIP model.
    """
    # This evaluation should only run on the main process
    if not is_master(args):
        return {}  # Return empty dict on non-master processes

    logger.info("--- Starting Image Retrieval Evaluation ---")

    # --- 1. Configuration ---
    benchmark_dir = args.retrieval_benchmark_dir
    all_paths_file = os.path.join(benchmark_dir, "all_paths.json")
    ground_truth_file = os.path.join(benchmark_dir, "ground_truth.json")

    # The root path for the image files, if they are not absolute
    # If your all_paths.json contains relative paths like 'images/001.jpg'
    # and your benchmark_dir is 'path/to/my_benchmark', set data_root to 'path/to/my_benchmark'
    data_root = args.retrieval_data_root or ""

    logger.info(f"Loading assets from: {benchmark_dir}")

    with open(all_paths_file, "r") as f:
        all_paths = json.load(f)
    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    # --- 2. Setup Dataset and Dataloader ---
    # Use the model's validation transform for consistency
    eval_transform = transform

    dataset = ImageDataset(all_paths, eval_transform, root_path=data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.retrieval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # --- 3. Compute Embeddings ---
    eval_model = unwrap_model(model)
    eval_model.eval()
    all_embeddings = []

    logger.info("Computing embeddings for the benchmark dataset...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Embeddings"):
            images = batch.to(args.device, non_blocking=True)

            # Use OpenCLIP's image encoder
            features = eval_model.encode_image(images)

            # Always L2-normalize for retrieval
            features = F.normalize(features, dim=-1)

            all_embeddings.append(features.cpu().numpy())

    master_embeddings = np.vstack(all_embeddings)

    # --- 4. FAISS Indexing and Search ---
    logger.info("Building FAISS index...")
    d = master_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    # For L2-normalized vectors, L2 distance is equivalent to cosine similarity
    index.add(master_embeddings.astype("float32"))
    logger.info(f"FAISS index built with {index.ntotal} vectors.")

    path_to_idx = {path: i for i, path in enumerate(all_paths)}
    query_paths = list(ground_truth.keys())

    # Handle cases where a query path might not be in all_paths
    query_indices = [path_to_idx[p] for p in query_paths if p in path_to_idx]
    if len(query_indices) != len(query_paths):
        logger.warning(
            f"Found {len(query_indices)} of {len(query_paths)} query images in the dataset."
        )
        query_paths = [
            p for p in query_paths if p in path_to_idx
        ]  # Filter query paths as well

    query_embeddings = master_embeddings[query_indices]

    recall_ks = args.retrieval_recall_ks
    search_k = max(recall_ks) + 1  # +1 to account for query-as-top-result
    logger.info(f"Performing batched search for top {search_k} results...")
    _, top_k_indices = index.search(query_embeddings.astype("float32"), search_k)

    # --- 5. Calculate Metrics ---
    recall_scores = {k: 0 for k in recall_ks}
    num_queries = len(query_paths)

    for i in tqdm(range(num_queries), desc="Calculating Recall@k"):
        query_path = query_paths[i]
        retrieved_indices = top_k_indices[i]
        retrieved_paths = [all_paths[idx] for idx in retrieved_indices if idx != -1]

        # Filter out the query itself from the results
        filtered_results = [p for p in retrieved_paths if p != query_path]

        true_positives = ground_truth[query_path]
        for k in recall_ks:
            if calculate_recall_at_k(filtered_results, true_positives, k):
                recall_scores[k] += 1

    # --- 6. Log and Prepare Results ---
    results = {}
    logger.info("--- Retrieval Benchmark Results ---")
    for k in recall_ks:
        score = (recall_scores[k] / num_queries) * 100
        logger.info(f"Recall@{k:<3}: {score:.2f}%")
        # Format for wandb/logging
        results[f"retrieval/Recall@{k}"] = score
    logger.info("-----------------------------------")

    # The main evaluate function will handle saving results.jsonl and wandb
    # We just return the metrics dictionary.
    return results
