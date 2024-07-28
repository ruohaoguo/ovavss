import os
import os.path as osp
import tempfile
import pickle
import shutil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

MAIN_RANK = 0


def _mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def is_enabled():
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return get_rank() == 0


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_enabled():
        return 1
    return dist.get_world_size()


def convert_model(model, **kwargs):
    if is_enabled():
        rank = get_rank()
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, **kwargs)
    else:
        model.cuda()
    return model


def unwrap_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def build_dataloader(dataset, **kwargs):
    if is_enabled():
        assert 'sampler' not in kwargs, 'Sampler can not be used in distributed mode!'
        shuffle = kwargs.get('shuffle', False)
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        sampler.set_epoch(1)
        batch_size = kwargs.get('batch_size', 1)
        world_size = get_world_size()
        assert batch_size % world_size == 0, 'Batch size must be divisible by world size!'
        kwargs['batch_size'] = batch_size // world_size
        kwargs['sampler'] = sampler
        kwargs['shuffle'] = False
    loader = DataLoader(dataset, **kwargs)
    return loader


def multi_process_run(func, args, nprocs=1, join=True, host='localhost', port='12355'):
    if nprocs == 1:
        func(0, *args)
    else:
        os.environ['MASTER_ADDR'] = host
        os.environ['MASTER_PORT'] = port
        mp.spawn(func, args=args, nprocs=nprocs, join=join)


def init_process_group(backend, world_size=1, rank=0):
    if world_size == 1:
        return
    dist.init_process_group(backend, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if dist.is_available() and dist.is_initialized():
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor


def reduce(tensor, dst=0, avg=True):
    if not is_enabled():
        return tensor
    dist.reduce(tensor, dst)
    tensor = tensor / get_world_size() if avg and is_main_process() else tensor
    return tensor


def reduce_dict(tensor_dict, dst=0, avg=True):
    if not is_enabled():
        return tensor_dict
    handle_dict = dict()
    for k, v in tensor_dict.items():
        reduce(v, dst)
        handle_dict[k] = v / get_world_size() if avg and is_main_process() else v
    return handle_dict


def all_reduce(tensor):
    if not is_enabled():
        return None
    return dist.all_reduce(tensor)


def all_reduce_dict(tensor_dict):
    if not is_enabled():
        return None
    handle_dict = dict()
    for k, v in tensor_dict.items():
        handle_dict[k] = all_reduce(v)
    return handle_dict


# def reduce(tensor, average=True):
#     world_size = dist.get_world_size()
#     if world_size < 2:
#         return tensor
#     with torch.no_grad():
#         dist.reduce(tensor, dst=0)
#         if dist.get_rank() == 0 and average:
#             # only main process gets accumulated, so only divide by
#             # world_size in this case
#             tensor = tensor / world_size
#     return tensor

# def reduce_dict(input_dict, average=True):
#     """
#     Reduce the values in the dictionary from all processes so that process with rank
#     0 has the reduced results.

#     Args:
#         input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
#         average (bool): whether to do average or sum

#     Returns:
#         a dict with the same keys as input_dict, after reduction.
#     """
#     if not is_enabled():
#         return input_dict
#     with torch.no_grad():
#         names = []
#         values = []
#         # sort the keys so that they are consistent across processes
#         for k in sorted(input_dict.keys()):
#             names.append(k)
#             values.append(input_dict[k])
#             dist.reduce(input_dict[k], dst=0)
#         if dist.get_rank() == 0 and average:
#             # only main process gets accumulated, so only divide by
#             # world_size in this case
#             values = [v / world_size for v in values]
#         reduced_dict = {k: v for k, v in zip(names, values)}
#     return reduced_dict

# def process_output(output_dict):
#     for k,v in output_dict.items():
#         if len(v.shape) == 0:


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.

    Returns:
        list: The collected results.
    """
    rank = get_rank()
    world_size = get_world_size()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            _mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        _mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    pickle.dump(result_part, open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_result = pickle.load(open(part_file, 'rb'))
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank = get_rank()
    world_size = get_world_size()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
