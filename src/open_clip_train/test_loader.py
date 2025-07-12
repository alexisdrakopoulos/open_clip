"""
This module purely exists to test the data loading functionality of the WebDataset loader
"""

from torchvision import transforms
from argparse import Namespace
import matplotlib.pyplot as plt
import logging

# Configure basic logging to see potential warnings from webdataset
logging.basicConfig(level=logging.INFO)

# Assuming data.py is in the same directory
try:
    from open_clip_train.data import get_wds_dataset
except ImportError:
    print("Error: Could not import from data.py. Make sure it's in the same directory.")
    exit()

# 1. Mock the arguments object that get_wds_dataset expects.
#    This simulates the configuration that would normally be passed during a training run.
# --- USER: PLEASE ADJUST THESE VALUES ---
args = Namespace(
    # --- Essential ---
    train_data="src/open_clip_train/unified-dataset-000609.tar",  # IMPORTANT: Change this to a real shard from your dataset
    dataset_type="webdataset",
    batch_size=4,
    workers=0,  # Using 0 workers simplifies debugging by avoiding multiprocessing.
    train_num_samples=3000,  # Adjust based on your shard size. This affects epoch length calculation.
    # --- Less critical for this test, but needed by the function signature ---
    val_data=None,
    dataset_resampled=False,
    seed=42,
    world_size=1,  # Assuming single GPU/CPU execution for this test
    train_data_upsampling_factors=None,
)
# --- END OF USER ADJUSTMENTS ---

# 2. Define a simple image preprocessing function.
#    This should ideally mirror your actual preprocessing to ensure the test is valid.
preprocess_img = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# 3. Define a placeholder tokenizer.
#    The data loader pipeline calls this on each caption. For this test, we
#    simply return the caption text as-is to print it.
def dummy_tokenizer(text):
    """
    In a real scenario, this would convert text to token IDs (e.g., a torch.Tensor).
    For inspection purposes, returning the raw text string is most useful.
    """
    return [text]


# 4. Get the DataInfo object by calling the function from data.py
print("Initializing WebDataset loader...")
# We are testing the training loader, so we set is_train=True
train_data_info = get_wds_dataset(
    args, preprocess_img=preprocess_img, is_train=True, tokenizer=dummy_tokenizer
)

# 5. Get the actual dataloader from the DataInfo object
dataloader = train_data_info.dataloader

# 6. Iterate through a few batches to inspect the data
num_batches_to_check = 5
print(f"--- Inspecting {num_batches_to_check} batches from the dataloader ---")

try:
    for i, batch in enumerate(dataloader):
        if i >= num_batches_to_check:
            break

        images, texts = batch

        print(f"\n--- Batch {i + 1} ---")
        # The images should be a tensor of shape [batch_size, channels, height, width]
        print(f"Images tensor shape: {images.shape}")
        print(f"Images tensor dtype: {images.dtype}")

        # The texts will be a list of strings because of our dummy_tokenizer
        print(f"Number of texts in batch: {len(texts)}")
        print("Captions pulled from dataset:")
        for j, text in enumerate(texts):
            # Limiting text print length for readability
            print(f"  Sample {j + 1}: {text}")

        # Optional: Display the first image of the batch for visual confirmation
        try:
            first_image_tensor = images[0]
            # To display a tensor, we need to convert it from (C, H, W) to (H, W, C)
            img_to_show = first_image_tensor.permute(1, 2, 0).numpy()
            plt.imshow(img_to_show)
            plt.title(f"Batch {i + 1}, First Image")
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"Could not display image. Matplotlib error: {e}")

except Exception as e:
    print(f"\nAn error occurred during data loading: {e}")
    print("Please check the following:")
    print(f"1. The path to your data shard is correct: '{args.train_data}'")
    print("2. The .tar file is not corrupted.")
    print(
        "3. The keys inside the .tar file ('image.jpg', 'data.json') match what's expected in data.py."
    )


print("\n--- Data loading check complete. ---")
