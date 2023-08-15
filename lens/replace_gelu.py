import onnx
import numpy as np
import sys, os
import torch

def create_initializer_scalar(
        name: str,
        scalar: float,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=[],
        vals=[scalar])

    return initializer_tensor

def create_initializer_tensor(
    name: str,
    tensor_array: np.ndarray,
    data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor

def make_const_tensor(
    name: str,
    values: np.ndarray,
    data_type: onnx.TensorProto
):
    const_tensor = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=[name],
        value=onnx.helper.make_tensor(
            name=name,
            data_type=data_type,
            dims=values.shape,
            vals=values.flatten().tolist(),
        ),
        name=name
    )
    return const_tensor

def make_mult_scalar_layer(
    nodes : list,
    input : str,
    scalar : float
):
    mult_scalar_const_node = make_const_tensor(
        name = "MulScalar_" + str(make_mult_scalar_layer.idx),
        values = np.array(scalar).astype(np.float32),
        data_type = onnx.TensorProto.FLOAT
    )
    nodes.append(mult_scalar_const_node)

    name = "Mult_" + str(make_mult_scalar_layer.idx)
    mult_scalar_node = onnx.helper.make_node(
        name=name,
        op_type="Mul",
        inputs=[input, mult_scalar_const_node.name],
        outputs=[name],
    )
    
    make_mult_scalar_layer.idx += 1
    return mult_scalar_node

make_mult_scalar_layer.idx = 0

def convertModel(path, res_path):
    """Replace the ReLU layers with SnapMl-compatible GELU layer
    """
    print(path, res_path)
    model = onnx.load(path)
    # onnx.checker.check_model(model)

    graph_def = model.graph
    initializers = graph_def.initializer
    inputs = graph_def.input
    outputs = graph_def.output
    nodes = graph_def.node

    for node in nodes:
        if (node.op_type == "Relu"):
            node.op_type = "GeluSnap"

    graph_def_2 = onnx.helper.make_graph(
        nodes=nodes,
        name=graph_def.name,
        inputs=inputs,  # Graph input
        outputs=outputs,  # Graph output
        initializer=initializers,
    )
    model_def = onnx.helper.make_model(graph_def_2, producer_name="onnx-example")
    model_def.opset_import[0].version = 11
    model_def = onnx.shape_inference.infer_shapes(model_def)
    # onnx.checker.check_model(model_def)
    onnx.save(model_def, res_path)
