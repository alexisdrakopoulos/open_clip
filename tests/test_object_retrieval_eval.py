import json
import types

import torch
from PIL import Image

from open_clip_train.data import get_object_retrieval_dataset, normalize_object_retrieval_path
from open_clip_train.train import get_object_retrieval_metrics


def _write_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (4, 4), color=color).save(path)


def test_object_retrieval_dataset_filters_ground_truth(tmp_path):
    data_root = tmp_path / 'clip_eval_set'
    assets_root = data_root / 'benchmark_assets'
    assets_root.mkdir(parents=True)

    paths = [
        'augmented_images/a/1.jpg',
        'augmented_images/a/2.jpg',
        'augmented_images/b/1.jpg',
    ]
    for index, path in enumerate(paths):
        _write_image(data_root / path, color=(index, index, index))

    with open(assets_root / 'all_paths.json', 'w', encoding='utf-8') as f:
        json.dump(paths + [paths[0]], f)
    with open(assets_root / 'ground_truth.json', 'w', encoding='utf-8') as f:
        json.dump({
            './augmented_images/a/1.jpg': [
                'augmented_images/a/2.jpg',
                'augmented_images/missing.jpg',
                'augmented_images/a/1.jpg',
            ],
            'augmented_images/missing-query.jpg': ['augmented_images/a/2.jpg'],
        }, f)

    args = types.SimpleNamespace(
        object_retrieval_root=str(data_root),
        object_retrieval_all_paths=None,
        object_retrieval_ground_truth=None,
        object_retrieval_max_images=0,
        object_retrieval_batch_size=2,
        batch_size=4,
        workers=0,
    )
    data_info = get_object_retrieval_dataset(args, lambda image: torch.zeros(3, 2, 2))

    assert data_info.paths == paths
    assert data_info.ground_truth == {'augmented_images/a/1.jpg': ['augmented_images/a/2.jpg']}
    batch = next(iter(data_info.dataloader))
    assert batch['image'].shape == (2, 3, 2, 2)
    assert batch['path'] == paths[:2]


def test_normalize_object_retrieval_path_handles_rooted_paths(tmp_path):
    data_root = tmp_path / 'clip_eval_set'
    path = data_root / 'augmented_images' / 'a.jpg'

    assert normalize_object_retrieval_path(str(path), str(data_root)) == 'augmented_images/a.jpg'
    assert normalize_object_retrieval_path('./augmented_images/a.jpg', str(data_root)) == 'augmented_images/a.jpg'


def test_object_retrieval_metrics_ignore_self_and_report_hit_recall():
    paths = [
        'object_a/query.jpg',
        'object_a/positive_near.jpg',
        'object_b/query.jpg',
        'object_a/positive_far.jpg',
        'object_b/positive.jpg',
    ]
    image_features = torch.tensor([
        [1.00, 0.00],
        [0.99, 0.01],
        [0.98, 0.02],
        [-1.00, 0.00],
        [0.00, 1.00],
    ])
    ground_truth = {
        'object_a/query.jpg': ['object_a/positive_near.jpg', 'object_a/positive_far.jpg'],
        'object_b/query.jpg': ['object_b/positive.jpg'],
    }

    metrics = get_object_retrieval_metrics(
        image_features=image_features,
        paths=paths,
        ground_truth=ground_truth,
        ks=[1, 2, 5],
        query_batch_size=1,
        device=torch.device('cpu'),
    )

    assert metrics['object_retrieval_samples'] == 5
    assert metrics['object_retrieval_queries'] == 2
    assert metrics['object_retrieval_positive_pairs'] == 3
    assert metrics['object_retrieval_recall@1'] == 0.5
    assert metrics['object_retrieval_recall@2'] == 0.5
    assert metrics['object_retrieval_recall@5'] == 1.0


def test_object_retrieval_metrics_accept_comma_separated_ks():
    paths = ['a/1.jpg', 'a/2.jpg']
    metrics = get_object_retrieval_metrics(
        image_features=torch.eye(2),
        paths=paths,
        ground_truth={'a/1.jpg': ['a/2.jpg']},
        ks=['1,5'],
        device=torch.device('cpu'),
    )

    assert metrics['object_retrieval_recall@1'] == 1.0
    assert metrics['object_retrieval_recall@5'] == 1.0