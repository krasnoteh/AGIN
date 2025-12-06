from collections import OrderedDict
import tensorrt as trt
import torch

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes


class Engine():
    def __init__(self, engine_path, dtype = torch.float16):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None
        self.dtype = dtype

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(tuple(shape), dtype=self.dtype).to(device=device)
            self.tensors[name] = tensor

    def infer(self, feed_dict, stream):

        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        noerror = self.context.execute_async_v3(stream.ptr)
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")

        return self.tensors