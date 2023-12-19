import os
import argparse
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import time

try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

def GiB(val):
    return val * 1 << 30

def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[]):
    '''
    Parses sample arguments.
    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.
    Returns:
        str: Path of data directory.
    Raises:
        FileNotFoundError
    '''

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory.", default=kDEFAULT_DATA_ROOT)
    args, unknown_args = parser.parse_known_args()

    # If data directory is not specified, use the default.
    data_root = args.datadir
    # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
    subfolder_path = os.path.join(data_root, subfolder)
    data_path = subfolder_path
    if not os.path.exists(subfolder_path):
        print("WARNING: " + subfolder_path + " does not exist. Trying " + data_root + " instead.")
        data_path = data_root

    # Make sure data directory exists.
    if not (os.path.exists(data_path)):
        raise FileNotFoundError(data_path + " does not exist. Please provide the correct data path with the -d option.")

    # Find all requested files.
    for index, f in enumerate(find_files):
        find_files[index] = os.path.abspath(os.path.join(data_path, f))
        if not os.path.exists(find_files[index]):
            raise FileNotFoundError(find_files[index] + " does not exist. Please provide the correct data path with the -d option.")

    return data_path, find_files

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, context, dynamic_shape={}, dynamic_shape_value={}):
    inputs = {}
    outputs = {}
    bindings = []
    stream = cuda.Stream()
    default_shape_dict = {
        "bbox": [100, 4],
        "score": [100],
        "pred_cls": [100]
    }
    for i, binding in enumerate(engine):
        # if binding in dynamic_shape_value:
        #     context.set_shape_input(i, dynamic_shape_value[binding])
        if binding in dynamic_shape:
            context.set_binding_shape(i, dynamic_shape[binding])
            # context.set_input_shape(binding, dynamic_shape[binding])
    for i, binding in enumerate(engine):
        size = trt.volume(context.get_binding_shape(i))
        if size == 0:
            default_shape = default_shape_dict[binding]
            size = trt.volume(default_shape)
        #   context.set_binding_shape(i, default_shape)
        # if context.all_binding_shapes_specified and context.all_shape_inputs_specified:
        #     print(binding, engine.get_binding_shape(binding), context.get_binding_shape(i))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs[i] = HostDeviceMem(host_mem, device_mem)
        else:
            outputs[i] = HostDeviceMem(host_mem, device_mem)
    return inputs, outputs, bindings, stream

def allocate_buffers_torch(engine, context, dynamic_shape={}, dynamic_shape_value={}, device=None):
    import torch
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    dtype_mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        # trt.uint8: torch.uint8,
        # Note: fp8 has no equivalent numpy type
    }
    for i, binding in enumerate(engine):
        if binding in dynamic_shape_value:
            context.set_shape_input(i, dynamic_shape_value[binding])
        if binding in dynamic_shape:
            context.set_binding_shape(i, dynamic_shape[binding])
    for i, binding in enumerate(engine):
        data_shape = [dim for dim in context.get_binding_shape(i)]
        trt_dtype = engine.get_binding_dtype(binding)
        torch_data = torch.zeros(data_shape, dtype=dtype_mapping[trt_dtype], device=device)
        if engine.binding_is_input(binding):
            inputs.append(torch_data)
        else:
            outputs.append(torch_data)
        bindings.append(torch_data.data_ptr())
    return inputs, outputs, bindings, stream

def do_inference_torch(context, bindings, stream=None, async_infer=False):
    if async_infer:
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    else:
        context.execute_v2(bindings=bindings)

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1, h2d=True, d2h=True, device_input={}):
    # Transfer input data to the GPU.

    if h2d:
        for i, inp in inputs.items():
            if i not in device_input:
                cuda.memcpy_htod_async(inp.device, inp.host, stream)
            else:
                cuda.memcpy_dtod_async(inp.device, device_input[i][0], device_input[i][1]*4, stream)

        # [cuda.memcpy_htod_async(inp.device, inp.host, stream) for i, inp in enumerate(inputs) if i not in device_indexes]
    # Run inference.
    if not d2h:
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Synchronize the stream
        stream.synchronize()
        # return [out.device for out in outputs]
    # Transfer predictions back from the GPU.
    else:
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs.values()]
        stream.synchronize()
    # Return only the host outputs.
        # return [out.host for out in outputs]
