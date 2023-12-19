import argparse
#from load_plugin_lib import load_plugin_lib
#load_plugin_lib()
import tensorrt as trt
import sys

def convert(src, dst):
    logger = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(logger, '')
    #registry = trt.get_plugin_registry()
    #plugin_creator = registry.get_plugin_creator("roi_align", "1", "")
    builder = trt.Builder(logger)
    profile = builder.create_optimization_profile()
    calib_profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    workspace = 20
    config.max_workspace_size = workspace * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(src):
        raise RuntimeError(f'failed to load ONNX file: {src}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    # dynamic_input = {
    #     "img": [(1,3,340,340), (1,3,800,1216), (1,3,1400,1400)]
    # }

    dynamic_input = {
        "image": [(3,340,340), (3,562,1000), (3,720,1080)]
    }
    # dynamic_input_value = {
    #     "img_shape": [(340,340,3), (768, 1344,3), (1400, 1400, 3)]
    # }
    for inp in inputs:
        if inp.name in dynamic_input:
            profile.set_shape(inp.name, *dynamic_input[inp.name])
            calib_profile.set_shape(inp.name, dynamic_input[inp.name][1], dynamic_input[inp.name][1], dynamic_input[inp.name][1])
        # if inp.name in dynamic_input_value:
        #     profile.set_shape_input(inp.name, *dynamic_input_value[inp.name])
        print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for oup in outputs:
        if oup.dtype == trt.DataType.INT8:
            oup.dynamic_range = (-127, 128)
    config.add_optimization_profile(profile)
    config.set_calibration_profile(calib_profile)
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    #fp32_layers = ["Conv_102", "Relu_103", "Conv_104", "Relu_105", "Conv_109", "Relu_110", "Conv_111", "Relu_112", "Conv_116", "Relu_117", "Conv_118", "Relu_119", "Conv_125", "Relu_126", "Conv_130", "Relu_131", "Conv_1321", "Relu_1322", "Conv_137", "Relu_138", "Conv_139", "Relu_140", "Conv_144", "Relu_145", "Conv_148", "Add_149", "Relu_150", "Conv_153", "Relu_154", "Conv_158", "Relu_159", "Conv_160", "Relu_161", "Conv_165", "Relu_166", "Conv_167", "Relu_168", "Conv_1711", "Relu_1712", "Conv_1713", "Relu_1714", "Conv_1716", "Conv_1719", "Relu_1720", "Conv_1721", "Relu_1722", "Conv_1723", "Add_1724", "Relu_1725", "Conv_1726", "Relu_1727", "Conv_1728", "Relu_1729", "Conv_174", "Relu_175", "Conv_176", "Add_177", "Relu_178", "Conv_1762", "Conv_1763", "Conv_1784", "Relu_1785", "Conv_1789", "Relu_1790", "Conv_179", "Relu_180", "Conv_1791", "Relu_1792", "Conv_181", "Relu_182", "Conv_183", "Add_184", "Relu_185", "Conv_1848", "Conv_1849", "Conv_186", "Relu_187", "Conv_1870", "Relu_1871", "Conv_1875", "Relu_1876", "Conv_1877", "Relu_1878", "Conv_19", "Relu_20", "Conv_193", "Relu_194", "Conv_195", "Relu_196", "Conv_200", "Relu_201", "Conv_202", "Relu_203", "Conv_207", "Relu_208", "Conv_209", "Relu_210", "Conv_211", "Add_212", "Relu_213", "Conv_214", "Relu_215", "Conv_22", "Relu_23", "Conv_228", "Relu_229", "Conv_24", "Relu_25", "Conv_247", "BatchNormalization_248", "Conv_26", "Conv_27", "Add_28", "Relu_29", "Conv_30", "Relu_31", "Conv_32", "Relu_33", "Conv_34", "Add_35", "Relu_36", "Conv_37", "Relu_38", "Conv_39", "Relu_40", "Conv_41", "Add_42", "Relu_43", "Conv_46", "Relu_47", "Conv_48", "Add_50", "Relu_51", "Conv_509", "Relu_510", "PWN(Add_534", "Add_558)", "Conv_559", "Relu_560", "Conv_618", "Relu_619", "Conv_63", "Add_64", "Relu_65", "Conv_70", "Add_71", "Relu_72", "Conv_75", "Relu_76", "Conv_77", "Add_79", "Relu_80", "Conv_78", "Conv_81", "Relu_82", "Conv_83", "Relu_84", "Conv_85", "Add_86", "Relu_87", "Conv_88", "Relu_89", "Conv_90", "Relu_91", "Conv_95", "Relu_96", "Conv_951", "BatchNormalization_952", "model_roi_heads_box_predictor_fc_1_weight_constant", "model_roi_heads_box_predictor_fc_2_weight_constant"]
    #for lay in layers:
    #    #if lay.name in fp32_layers:
    #    #    lay.precision = trt.float32
    #    if lay.name.startswith("onnx::Add")
    #    if lay.type in [trt.LayerType.CONSTANT, trt.LayerType.SOFTMAX, trt.LayerType.ACTIVATION, trt.LayerType.NORMALIZATION]:
    #        if lay.type == trt.LayerType.CONSTANT and lay.get_output_type(0) == trt.DataType.FLOAT:
    #            lay.precision = trt.float32
    #        elif lay.type != trt.LayerType.CONSTANT:
    #            lay.precision = trt.float32
    for out in outputs:
        print(f'output "{out.name}" with shape{out.shape} {out.dtype}')
    half = False
    print(f'building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine')
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(dst, 'wb') as t:
        t.write(engine.serialize())


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src pytorch model path')
    parser.add_argument('dst', help='dst onnx model path')

    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
