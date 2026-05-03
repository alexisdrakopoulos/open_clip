import json
import logging
import math
from collections import Counter

_logger = logging.getLogger(__name__)
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip.task import get_model_from_task
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(task, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(
        args.precision,
        device_type=device.type,
        fsdp=getattr(args, 'fsdp', False),
    )
    input_dtype = get_input_dtype(args.precision)

    task.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_batches, accum_features = [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    is_master_rank = is_master(args)
    progress = tqdm(
        total=num_batches_per_epoch,
        desc=f"Train Epoch: {epoch}",
        unit="batch",
        dynamic_ncols=True,
        disable=not is_master_rank,
    )
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        batch = task.prepare_batch(batch, device=device, input_dtype=input_dtype)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                losses = task(batch)
                total_loss = losses["loss"]

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = task.trainable_module(**batch)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_batches.append(batch)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                batch_j = accum_batches[j]

                # Disable gradient sync for all but the last accumulation step.
                # FSDP2: set_requires_gradient_sync; DDP: no_sync context manager.
                is_last_step = (j == args.accum_freq - 1)
                use_fsdp_no_sync = (
                    not is_last_step
                    and hasattr(task.trainable_module, 'set_requires_gradient_sync')
                )
                use_ddp_no_sync = (
                    not is_last_step
                    and not use_fsdp_no_sync
                    and isinstance(task.trainable_module, DistributedDataParallel)
                )
                if use_fsdp_no_sync:
                    task.trainable_module.set_requires_gradient_sync(False)

                ddp_context = task.trainable_module.no_sync() if use_ddp_no_sync else nullcontext()
                with ddp_context:
                    with autocast():
                        model_out = task.trainable_module(**batch_j)

                        inputs_no_accum = {}
                        inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                        if "logit_bias" in model_out:
                            inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                        inputs = {}
                        for key, val in accum_features.items():
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                        losses = task.compute_accum_loss(inputs, inputs_no_accum, accum_batches)
                        del inputs
                        del inputs_no_accum
                        total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
                        losses["loss"] = total_loss
                        losses["logit_scale"] = logit_scale

                    backward(total_loss, scaler)

                if use_fsdp_no_sync:
                    task.trainable_module.set_requires_gradient_sync(True)

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    task.trainable_module.parameters(), args.grad_clip_norm, norm_type=2.0,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    task.trainable_module.parameters(), args.grad_clip_norm, norm_type=2.0,
                )
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_batches, accum_features = [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        task.clamp_logit_scale()

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master_rank:
            progress.update(1)

        if is_master_rank and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(batch["image"])
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale = losses.get("logit_scale", None)
            logit_scale_scalar = logit_scale.item() if logit_scale is not None else 0.0
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            progress.set_postfix_str(
                f"samples {num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%), "
                f"data {data_time_m.avg:.3f}s, "
                f"batch {batch_time_m.avg:.3f}s, "
                f"{samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu, "
                f"lr {optimizer.param_groups[0]['lr']:5f}, "
                f"scale {logit_scale_scalar:.3f}, " + loss_log,
                refresh=True,
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
            progress.close()
    # end for


def _metadata_to_list(values):
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def _metadata_to_string(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _index_batch_value(value, indices):
    if isinstance(value, torch.Tensor):
        index = torch.as_tensor(indices, device=value.device, dtype=torch.long)
        return value.index_select(0, index)
    if isinstance(value, dict):
        return {key: _index_batch_value(val, indices) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [value[index] for index in indices]
    return value


def _select_culture_knn_batch(batch, args, limit=None):
    cultures = [_metadata_to_string(value) for value in _metadata_to_list(batch.get("culture"))]
    if not cultures:
        return None, []

    include_unknown = getattr(args, 'culture_knn_include_unknown', False)
    selected_indices, selected_cultures = [], []
    for index, culture in enumerate(cultures):
        if not culture:
            continue
        if not include_unknown and culture.upper() == "UNKNOWN":
            continue
        selected_indices.append(index)
        selected_cultures.append(culture)

    if limit is not None:
        selected_indices = selected_indices[:limit]
        selected_cultures = selected_cultures[:limit]

    if not selected_cultures:
        return None, []

    return {
        key: _index_batch_value(value, selected_indices)
        for key, value in batch.items()
    }, selected_cultures


def _extract_image_features(model_out):
    if isinstance(model_out, dict):
        return model_out["image_features"]
    return model_out[0]


def _knn_device(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return device


def _knn_full_matrix_max_bytes(device):
    if device.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            return int(free_bytes * 0.6)
        except RuntimeError:
            return 4 * 1024 ** 3
    if device.type == "mps":
        return 2 * 1024 ** 3
    return 4 * 1024 ** 3


def _knn_similarity_matrix_bytes(num_queries, num_references):
    return num_queries * num_references * 4


def _auto_knn_query_batch_size(num_samples, device):
    max_bytes = max(256 * 1024 ** 2, _knn_full_matrix_max_bytes(device) // 2)
    return max(1, min(num_samples, max_bytes // max(1, num_samples * 4)))


def _culture_knn_counts(neighbor_labels, target_labels):
    target_labels = target_labels.unsqueeze(1)
    matches = neighbor_labels.eq(target_labels)
    majority_labels = torch.mode(neighbor_labels, dim=1).values
    return (
        matches[:, 0].float().sum().item(),
        matches.float().mean(dim=1).sum().item(),
        majority_labels.eq(target_labels.squeeze(1)).float().sum().item(),
    )


def get_culture_knn_metrics(
        image_features,
        cultures,
        k=5,
        query_batch_size=0,
        device=None,
        use_tqdm=False,
):
    num_samples = image_features.shape[0]
    class_counts = Counter(cultures)
    metrics = {
        "culture_knn_samples": num_samples,
        "culture_knn_classes": len(class_counts),
    }
    if num_samples < 2 or not class_counts:
        return metrics

    effective_k = min(max(1, int(k)), num_samples - 1)
    majority_baseline = max(class_counts.values()) / num_samples

    label_to_index = {label: index for index, label in enumerate(class_counts.keys())}
    label_indices = torch.tensor([label_to_index[label] for label in cultures], dtype=torch.long)

    image_features = F.normalize(image_features.float(), dim=-1)
    if device is None:
        device = torch.device("cpu")
    image_features = image_features.to(device=device, non_blocking=True)
    label_indices = label_indices.to(device=device, non_blocking=True)
    reference_features = image_features.t()

    top1_correct = 0.0
    same_at_k_sum = 0.0
    majority_correct = 0.0
    full_matrix_bytes = _knn_similarity_matrix_bytes(num_samples, num_samples)
    if full_matrix_bytes <= _knn_full_matrix_max_bytes(device):
        progress = tqdm(
            total=1,
            desc="Culture KNN search",
            unit="matrix",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        similarities = image_features @ reference_features
        similarities.fill_diagonal_(-float("inf"))
        neighbors = similarities.topk(effective_k, dim=1).indices
        del similarities
        neighbor_labels = label_indices[neighbors]
        top1_correct, same_at_k_sum, majority_correct = _culture_knn_counts(
            neighbor_labels,
            label_indices,
        )
        progress.update(1)
        progress.close()
    else:
        query_batch_size = int(query_batch_size)
        if query_batch_size <= 0:
            query_batch_size = _auto_knn_query_batch_size(num_samples, device)
        query_batch_size = max(1, min(query_batch_size, num_samples))
        _logger.info(
            "Culture KNN full matrix would require %.2f GiB; using exact chunked search with query_batch_size=%d.",
            full_matrix_bytes / (1024 ** 3),
            query_batch_size,
        )
        query_starts = range(0, num_samples, query_batch_size)
        if use_tqdm:
            query_starts = tqdm(
                query_starts,
                total=math.ceil(num_samples / query_batch_size),
                desc="Culture KNN search",
                unit="chunk",
                dynamic_ncols=True,
            )
        for start in query_starts:
            end = min(start + query_batch_size, num_samples)
            similarities = image_features[start:end] @ reference_features
            row_index = torch.arange(end - start, device=device)
            col_index = torch.arange(start, end, device=device)
            similarities[row_index, col_index] = -float("inf")

            neighbors = similarities.topk(effective_k, dim=1).indices
            neighbor_labels = label_indices[neighbors]
            batch_top1, batch_same_at_k, batch_majority = _culture_knn_counts(
                neighbor_labels,
                label_indices[start:end],
            )
            top1_correct += batch_top1
            same_at_k_sum += batch_same_at_k
            majority_correct += batch_majority

    metrics.update({
        "culture_knn_k": effective_k,
        "culture_knn_top1": top1_correct / num_samples,
        f"culture_knn_same@{effective_k}": same_at_k_sum / num_samples,
        f"culture_knn_majority@{effective_k}": majority_correct / num_samples,
        "culture_knn_majority_baseline": majority_baseline,
    })
    return metrics


def culture_knn_eval(task, data, epoch, args):
    if not getattr(args, 'culture_knn', False) or 'culture-knn' not in data:
        return {}

    frequency = getattr(args, 'culture_knn_frequency', 1)
    if frequency == 0:
        return {}
    if epoch != 0 and epoch != args.epochs and (epoch % frequency) != 0:
        return {}

    use_fsdp_eval = getattr(args, 'fsdp', False) and getattr(args, 'distributed', False)
    is_rank0 = is_master(args)
    if not use_fsdp_eval and not is_rank0:
        return {}

    device = torch.device(args.device)
    autocast = get_autocast(
        args.precision,
        device_type=device.type,
        fsdp=getattr(args, 'fsdp', False),
    )
    input_dtype = get_input_dtype(args.precision)
    model = get_model_from_task(task)
    knn_device = _knn_device(args)
    max_samples = max(0, getattr(args, 'culture_knn_max_samples', 0))

    all_image_features, all_cultures = [], []
    num_samples = 0
    progress = None
    if is_rank0:
        dataloader = data['culture-knn'].dataloader
        dataloader_iter = iter(dataloader)
        split = getattr(data['culture-knn'], 'split', 'dataset')
        _logger.info(f"Starting culture KNN eval on {split} data.")
        num_batches = getattr(dataloader, 'num_batches', None) or None
        progress = tqdm(
            total=num_batches,
            desc=f"Culture KNN embeddings ({split})",
            unit="batch",
            dynamic_ncols=True,
        )

    if use_fsdp_eval:
        image_size = model.visual.image_size
        if not isinstance(image_size, tuple):
            image_size = (image_size, image_size)
        dummy_images = torch.zeros(1, 3, *image_size, device=device, dtype=input_dtype)
        signal = torch.zeros(1, device=device, dtype=torch.long)

    with torch.inference_mode():
        i = 0
        while True:
            if use_fsdp_eval:
                if is_rank0:
                    if max_samples and num_samples >= max_samples:
                        batch = None
                    else:
                        batch = next(dataloader_iter, None)
                    if batch is not None and progress is not None:
                        progress.update(1)
                    signal.fill_(0 if batch is None else 1)
                dist.broadcast(signal, src=0)
                if signal.item() == 0:
                    break

                batch_cultures = []
                if is_rank0:
                    remaining = max_samples - num_samples if max_samples else None
                    batch, batch_cultures = _select_culture_knn_batch(batch, args, limit=remaining)
                    if batch is not None:
                        model_batch = task.prepare_batch({"image": batch["image"]}, device, input_dtype)
                    else:
                        model_batch = {"image": dummy_images}
                else:
                    model_batch = {"image": dummy_images}
            else:
                batch = next(dataloader_iter, None)
                if batch is None:
                    break
                if progress is not None:
                    progress.update(1)
                remaining = max_samples - num_samples if max_samples else None
                if remaining is not None and remaining <= 0:
                    break
                batch, batch_cultures = _select_culture_knn_batch(batch, args, limit=remaining)
                if batch is None:
                    i += 1
                    continue
                model_batch = task.prepare_batch({"image": batch["image"]}, device, input_dtype)

            with autocast():
                model_out = task(model_batch)

            if is_rank0 and batch_cultures:
                image_features = _extract_image_features(model_out)[:len(batch_cultures)]
                image_features = F.normalize(image_features.float(), dim=-1)
                image_features = image_features.to(device=knn_device, non_blocking=True)
                all_image_features.append(image_features)
                all_cultures.extend(batch_cultures)
                num_samples += len(batch_cultures)
                if progress is not None:
                    progress.set_postfix(samples=num_samples, refresh=False)
                if (i % 100) == 0:
                    _logger.info(f"Culture KNN Eval Epoch: {epoch} [{num_samples} samples]")
            i += 1

    if progress is not None:
        progress.close()

    if not is_rank0:
        return {}

    if not all_image_features:
        _logger.warning('Culture KNN eval found no samples after source/culture filtering.')
        return {"culture_knn_samples": 0, "culture_knn_classes": 0}

    image_features = torch.cat(all_image_features)
    metrics = get_culture_knn_metrics(
        image_features=image_features,
        cultures=all_cultures,
        k=getattr(args, 'culture_knn_k', 5),
        query_batch_size=getattr(args, 'culture_knn_query_batch_size', 1024),
        device=knn_device,
        use_tqdm=True,
    )
    _logger.info('Finished culture KNN eval.')
    return metrics


def evaluate(task, data, epoch, args, tb_writer=None, tokenizer=None):
    """Run validation + zero-shot eval. ``task`` must be a TrainingTask subclass.

    The image+text-shaped val loop below assumes an ImageTextTask (or compiled
    wrapper around one); other modalities will need their own eval entry point.
    """
    metrics = {}
    use_fsdp_eval = getattr(args, 'fsdp', False) and getattr(args, 'distributed', False)
    is_rank0 = is_master(args)

    if not use_fsdp_eval and not is_rank0:
        return metrics

    device = torch.device(args.device)
    task.eval()

    model = get_model_from_task(task)

    zero_shot_metrics = zero_shot_eval(task, data, epoch, args, tokenizer=tokenizer)
    if is_rank0:
        metrics.update(zero_shot_metrics)

    culture_knn_metrics = culture_knn_eval(task, data, epoch, args)
    if is_rank0:
        metrics.update(culture_knn_metrics)

    autocast = get_autocast(
        args.precision,
        device_type=device.type,
        fsdp=getattr(args, 'fsdp', False),
    )
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        num_samples = 0
        samples_per_val = 0

        if is_rank0:
            dataloader = data['val'].dataloader
            samples_per_val = dataloader.num_samples
            dataloader_iter = iter(dataloader)

        if use_fsdp_eval:
            # Pre-allocate dummy batch for non-master ranks
            dummy_batch = task.create_dummy_batch(
                image_size=model.visual.image_size,
                context_length=model.context_length,
                batch_size=1,
                device=device,
                dtype=input_dtype,
            )
            signal = torch.zeros(1, device=device, dtype=torch.long)

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            i = 0
            while True:
                if use_fsdp_eval:
                    if is_rank0:
                        batch = next(dataloader_iter, None)
                        signal.fill_(0 if batch is None else 1)
                    dist.broadcast(signal, src=0)
                    if signal.item() == 0:
                        break

                    if is_rank0:
                        batch = task.prepare_batch(batch, device, input_dtype)
                    else:
                        batch = dummy_batch
                else:
                    batch = next(dataloader_iter, None)
                    if batch is None:
                        break
                    batch = task.prepare_batch(batch, device, input_dtype)

                with autocast():
                    model_out = task(batch)

                if is_rank0:
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = len(batch["image"])
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out, texts=batch.get("text"))

                    cumulative_loss += total_loss * batch_size
                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                    num_samples += batch_size
                    if (i % 100) == 0:
                        _logger.info(
                            f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                            f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                        if gen_loss is not None:
                            _logger.info(
                                f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

                i += 1

            if is_rank0 and num_samples > 0:
                val_metrics = get_clip_metrics(
                    image_features=torch.cat(all_image_features),
                    text_features=torch.cat(all_text_features),
                    logit_scale=logit_scale.cpu(),
                )
                loss = cumulative_loss / num_samples
                metrics.update(
                    {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
                )
                if gen_loss is not None:
                    gen_loss = cumulative_gen_loss / num_samples
                    metrics.update({"val_generative_loss": gen_loss.item()})

    if not is_rank0:
        return metrics

    if not metrics:
        return metrics

    _logger.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out, texts=None, pad_id=0):
    if "logits" in model_out and texts is not None:
        logits = model_out["logits"][:, :-1]
        labels = texts[:, 1:]
        return F.cross_entropy(logits.permute(0, 2, 1), labels, ignore_index=pad_id)
