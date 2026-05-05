import ast
import json
import logging
import math

_logger = logging.getLogger(__name__)
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample



class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        _logger.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        # Keep as pandas Series rather than converting to Python lists.
        # Python lists of strings cause copy-on-write memory duplication in
        # forked DataLoader workers because the cyclic GC walks every element
        # and modifies reference counts.  Pandas Series are backed by
        # numpy/pyarrow buffers that the GC does not traverse.
        self.images = df[img_key]
        self.captions = df[caption_key]
        self.transforms = transforms
        _logger.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transforms(Image.open(str(self.images.iloc[idx])))
        text = self.tokenize([str(self.captions.iloc[idx])])[0]
        return {"image": image, "text": text}


def normalize_object_retrieval_path(path, data_root):
    path = str(path).strip().replace('\\', '/')
    if os.path.isabs(path):
        abs_root = os.path.abspath(data_root).replace(os.sep, '/').rstrip('/')
        abs_path = os.path.abspath(path).replace(os.sep, '/')
        root_prefix = abs_root + '/'
        if abs_path.startswith(root_prefix):
            return abs_path[len(root_prefix):]
        return abs_path
    while path.startswith('./'):
        path = path[2:]
    return path


def resolve_object_retrieval_path(data_root, path):
    return path if os.path.isabs(path) else os.path.join(data_root, path)


def _load_object_retrieval_json(path, expected_type):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, expected_type):
        raise ValueError(f"Expected {path} to contain {expected_type.__name__}.")
    return data


def _unique_preserve_order(values):
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


class ObjectRetrievalDataset(Dataset):
    def __init__(self, data_root, paths, transform):
        self.data_root = data_root
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image_path = resolve_object_retrieval_path(self.data_root, path)
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        return {"image": image, "path": path}


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for batch in dataloader:
        n_batches += 1
        n_elements += len(batch["image"])
        assert len(batch["image"]) == len(batch["text"])
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def decode_text_value(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


class TokenizeTextTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        return self.tokenizer(text)[0]


def filter_has_culture_and_source(sample):
    return 'culture' in sample and 'source' in sample


class SourceFilter:
    def __init__(self, sources):
        self.sources = set(sources)

    def __call__(self, sample):
        return decode_text_value(sample.get('source')) in self.sources


def make_source_filter(sources):
    if not sources:
        return None
    source_set = {decode_text_value(source) for source in sources if decode_text_value(source)}
    if not source_set:
        return None
    return SourceFilter(source_set)


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    _logger.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(
        args,
        preprocess_img,
        is_train,
        epoch=0,
        floor=False,
        tokenizer=None,
        input_shards=None,
        include_metadata=False,
        include_text=True,
        source_filter=None,
):
    input_shards = input_shards or (args.train_data if is_train else args.val_data)
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train
    source_filter_fn = make_source_filter(source_filter)
    include_metadata = include_metadata or source_filter_fn is not None

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.append(wds.select(filter_no_caption_or_no_image))
    if include_metadata:
        pipeline.append(wds.select(filter_has_culture_and_source))
    if source_filter_fn is not None:
        pipeline.append(wds.select(source_filter_fn))

    rename_kwargs = {"image": "jpg;png;jpeg;webp"}
    map_kwargs = {"image": preprocess_img}
    if include_text:
        rename_kwargs["text"] = "txt"
        map_kwargs["text"] = TokenizeTextTransform(tokenizer)
    if include_metadata:
        rename_kwargs.update({"culture": "culture", "source": "source"})
        map_kwargs.update({"culture": decode_text_value, "source": decode_text_value})

    pipeline.extend([
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(**rename_kwargs, keep=False),
        wds.map_dict(**map_kwargs),
        wds.batched(args.batch_size, partial=not is_train, collation_fn=default_collate),
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = TokenizeTextTransform(tokenizer)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return {"image": image, "text": self.preprocess_txt(self.caption)}


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_culture_knn_sources(args):
    sources = []
    for value in getattr(args, 'culture_knn_sources', None) or []:
        sources.extend(source.strip() for source in value.split(',') if source.strip())

    sources_file = getattr(args, 'culture_knn_sources_file', None)
    if sources_file:
        with open(sources_file, 'r', encoding='utf-8') as f:
            for line in f:
                source = line.strip()
                if source and not source.startswith('#'):
                    sources.append(source)

    return sources or None


def get_culture_knn_data_path(args):
    split = getattr(args, 'culture_knn_data', 'auto')
    if split == 'val':
        return args.val_data, 'val'
    if split == 'train':
        return args.train_data, 'train'
    if args.val_data:
        return args.val_data, 'val'
    return args.train_data, 'train'


def get_object_retrieval_dataset(args, preprocess_fn):
    data_root = getattr(args, 'object_retrieval_root', None)
    if not data_root:
        raise ValueError('--object-retrieval-root is required when --object-retrieval is enabled.')

    assets_root = os.path.join(data_root, 'benchmark_assets')
    all_paths_path = getattr(args, 'object_retrieval_all_paths', None) or os.path.join(assets_root, 'all_paths.json')
    ground_truth_path = getattr(args, 'object_retrieval_ground_truth', None) or os.path.join(assets_root, 'ground_truth.json')

    all_paths = _load_object_retrieval_json(all_paths_path, list)
    ground_truth = _load_object_retrieval_json(ground_truth_path, dict)

    paths = _unique_preserve_order(
        normalize_object_retrieval_path(path, data_root)
        for path in all_paths
    )
    max_images = max(0, int(getattr(args, 'object_retrieval_max_images', 0)))
    if max_images:
        paths = paths[:max_images]

    available_paths = set(paths)
    filtered_ground_truth = {}
    for query_path, positive_paths in ground_truth.items():
        query_path = normalize_object_retrieval_path(query_path, data_root)
        if query_path not in available_paths:
            continue
        positives = []
        for positive_path in positive_paths:
            positive_path = normalize_object_retrieval_path(positive_path, data_root)
            if positive_path != query_path and positive_path in available_paths:
                positives.append(positive_path)
        positives = _unique_preserve_order(positives)
        if positives:
            filtered_ground_truth[query_path] = positives

    if not paths:
        _logger.warning('Object retrieval eval found no image paths in %s.', all_paths_path)
    if not filtered_ground_truth:
        _logger.warning('Object retrieval eval found no usable queries in %s.', ground_truth_path)

    dataset = ObjectRetrievalDataset(data_root, paths, preprocess_fn)
    batch_size = max(1, int(getattr(args, 'object_retrieval_batch_size', 0) or args.batch_size))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    data_info = DataInfo(dataloader=dataloader)
    data_info.ground_truth = filtered_ground_truth
    data_info.paths = paths
    data_info.root = data_root
    return data_info


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if getattr(args, 'culture_knn', False):
        culture_knn_path, culture_knn_split = get_culture_knn_data_path(args)
        if not culture_knn_path:
            _logger.warning('Skipping culture KNN eval because no matching train/val WebDataset path was provided.')
        else:
            culture_knn_dataset_fn = get_dataset_fn(culture_knn_path, args.dataset_type)
            if culture_knn_dataset_fn is not get_wds_dataset:
                _logger.warning('Skipping culture KNN eval because it currently supports WebDataset tar shards only.')
            else:
                data["culture-knn"] = get_wds_dataset(
                    args,
                    preprocess_val,
                    is_train=False,
                    epoch=epoch,
                    tokenizer=tokenizer,
                    input_shards=culture_knn_path,
                    include_metadata=True,
                    include_text=False,
                    source_filter=get_culture_knn_sources(args),
                )
                data["culture-knn"].split = culture_knn_split

    if getattr(args, 'object_retrieval', False):
        data["object-retrieval"] = get_object_retrieval_dataset(args, preprocess_val)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
