"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist_1(devices=(0,)):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    try:
        device_string = ','.join(map(str, devices))
    except TypeError:
        device_string = str(devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_string #f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
    print('device_string', device_string)
    print('os.environ["CUDA_VISIBLE_DEVICES"]', os.environ["CUDA_VISIBLE_DEVICES"])
    # comm = MPI.COMM_WORLD
    # print('commworld, 'f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}", comm)
    backend = "gloo" if not th.cuda.is_available() else "nccl"
   # print('commrank', comm.rank)
   # print('commsize', comm.size)

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = '127.0.1.1'#comm.bcast(hostname, root=0)
    os.environ["RANK"] = '0'#str(comm.rank)
    os.environ["WORLD_SIZE"] = '1'#str(comm.size)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    print('port2', port)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")
    print('dist init done')

def setup_dist_n(devices=(0,)):
    """
    Setup a distributed process group.
    参数 devices: 可以是单个整数 (如 3), 也可以是列表/元组 (如 [0, 1])
    """
    # [关键检查] 防止在设置显卡前 CUDA 已经被初始化 (会导致设置失效)
    if th.cuda.is_initialized():
        print(f"⚠️ Warning: CUDA has already been initialized! Setting CUDA_VISIBLE_DEVICES={devices} might NOT work.")
        print("Please ensure setup_dist() is called before any torch.cuda functions.")

    if dist.is_initialized():
        return

    # [逻辑优化] 解析输入设备，将其转换为字符串 (例如: "0,1" 或 "3")
    if isinstance(devices, (list, tuple)):
        device_string = "{},{}".format(devices[0], devices[-1])
    else:
        device_string = str(devices)

    # [核心设置] 设置物理可见显卡
    # 注意: 一旦设置了 "3", 程序内部看到的将是 "cuda:0", 但实际跑在物理 GPU 3 上
    os.environ["CUDA_VISIBLE_DEVICES"] = device_string

    print(f"🚀 Setup Dist: Setting CUDA_VISIBLE_DEVICES = '{device_string}'")
    # print('os.environ["CUDA_VISIBLE_DEVICES"]', os.environ["CUDA_VISIBLE_DEVICES"])

    # 自动选择后端
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    # 设置 Master Address (单机模式下固定为本地)
    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = "127.0.0.1"  # socket.gethostbyname(socket.getfqdn())

    os.environ["MASTER_ADDR"] = hostname

    # 伪分布式设置 (单进程模式)
    os.environ["RANK"] = '0'  # 永远是 0 号进程
    os.environ["WORLD_SIZE"] = '1'  # 总共只有 1 个进程

    # 动态寻找可用端口，防止端口冲突
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()

    print(f"🚀 Setup Dist: Using Master Port = {port}")
    os.environ["MASTER_PORT"] = str(port)

    # 初始化进程组
    dist.init_process_group(backend=backend, init_method="env://")
    print('✅ dist init done')


def setup_dist(devices=(0,)):
    if len(devices)==1:
        setup_dist_1( devices)
    elif len(devices)>1:
        setup_dist_n(devices)

def dev(device_number=0):
    """
    Get the device to use for torch.distributed.
    Modified to ALWAYS return specific device indices (e.g., cuda:0) to avoid mismatch errors.
    """
    if isinstance(device_number, (list, tuple)):
        return [dev(k) for k in device_number]

    if th.cuda.is_available():
        device_count = th.cuda.device_count()

        # [修改] 删除了 if device_count == 1 的特殊判断
        # 无论有几张卡，只要请求的编号合法，都返回具体的 'cuda:x'
        if device_number < device_count:
            return th.device(f'cuda:{device_number}')
        else:
            raise ValueError(
                f'requested device number {device_number} (0-indexed) but only {device_count} devices available')

    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    #print('mpicommworldgetrank', MPI.COMM_WORLD.Get_rank())
    mpigetrank=0
   # if MPI.COMM_WORLD.Get_rank() == 0:
    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
   # data = MPI.COMM_WORLD.bcast(data)
  #  print('mpibacst', MPI.COMM_WORLD.bcast(data))
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    #for p in params:
    #    with th.no_grad():
    #        dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
