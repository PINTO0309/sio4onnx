#! /usr/bin/env python

import sys
import onnx
from onnx import ModelProto, ValueInfoProto, TensorShapeProto
from typing import Optional, List, Union, Any
from argparse import ArgumentParser
from ast import literal_eval

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'


def update_inputs_outputs_dims(
    model: ModelProto,
    input_dims: dict[str, list[Any]],
    output_dims: dict[str, list[Any]],
) -> ModelProto:
    """This function updates the dimension sizes of the model's inputs and outputs to the values
    provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
    will be set for that dimension.

    Example. if we have the following shape for inputs and outputs:

    * shape(input_1) = ('b', 3, 'w', 'h')
    * shape(input_2) = ('b', 4)
    * shape(output)  = ('b', 'd', 5)

    The parameters can be provided as:

    ::

        input_dims = {
            "input_1": ['b', 3, 'w', 'h'],
            "input_2": ['b', 4],
        }
        output_dims = {
            "output": ['b', -1, 5]
        }

    Putting it together:

    ::

        model = onnx.load('model.onnx')
        updated_model = update_inputs_outputs_dims(model, input_dims, output_dims)
        onnx.save(updated_model, 'model.onnx')
    """
    dim_param_set: set[str] = set()

    def init_dim_param_set(
        dim_param_set: set[str], value_infos: list[ValueInfoProto]
    ) -> None:
        for info in value_infos:
            shape = info.type.tensor_type.shape
            for dim in shape.dim:
                if dim.HasField("dim_param"):
                    dim_param_set.add(dim.dim_param)

    init_dim_param_set(dim_param_set, model.graph.input)
    init_dim_param_set(dim_param_set, model.graph.output)
    init_dim_param_set(dim_param_set, model.graph.value_info)

    def update_dim(tensor: ValueInfoProto, dim: Any, j: int, name: str) -> None:
        dim_proto = tensor.type.tensor_type.shape.dim[j]
        if isinstance(dim, int):
            if dim >= 0:
                if dim_proto.HasField("dim_value") and dim_proto.dim_value != dim:
                    raise ValueError(
                        f"Unable to set dimension value to {dim} for axis {j} of {name}. Contradicts existing dimension value {dim_proto.dim_value}."
                    )
                dim_proto.dim_value = dim
            else:
                generated_dim_param = name + "_" + str(j)
                if generated_dim_param in dim_param_set:
                    raise ValueError(
                        f"Unable to generate unique dim_param for axis {j} of {name}. Please manually provide a dim_param value."
                    )
                dim_proto.dim_param = generated_dim_param
        elif isinstance(dim, str):
            dim_proto.dim_param = dim
        else:
            raise ValueError(
                f"Only int or str is accepted as dimension value, incorrect type: {type(dim)}"
            )

    def make_dim(tensor: ValueInfoProto, output_dim_arr: Any, name: str) -> None:
        make_dim_list = []
        for j, dim in enumerate(output_dim_arr):
            if isinstance(dim, int):
                if dim >= 0:
                    make_dim = TensorShapeProto.Dimension(dim_value=dim)
                    make_dim_list.append(make_dim)
                else:
                    make_dim = TensorShapeProto.Dimension(dim_param=str(dim))
                    make_dim_list.append(make_dim)
            elif isinstance(dim, str):
                make_dim = TensorShapeProto.Dimension(dim_param=dim)
                make_dim_list.append(make_dim)
            else:
                raise ValueError(
                    f"Only int or str is accepted as dimension value, incorrect type: {type(dim)}"
                )
        make_tensor_shape_proto = TensorShapeProto(dim=make_dim_list)
        tensor.type.tensor_type.shape.MergeFrom(make_tensor_shape_proto)

    for input_ in model.graph.input:
        input_name = input_.name
        input_dim_arr = input_dims[input_name]

        if input_.type.tensor_type.shape.dim != []:
            for j, dim in enumerate(input_dim_arr):
                update_dim(input_, dim, j, input_name)
        else:
            make_dim(input_, input_dim_arr, input_name)

    for output in model.graph.output:
        output_name = output.name
        output_dim_arr = output_dims[output_name]

        if output.type.tensor_type.shape.dim != []:
            for j, dim in enumerate(output_dim_arr):
                update_dim(output, dim, j, output_name)
        else:
            make_dim(output, output_dim_arr, output_name)

    onnx.checker.check_model(model)
    return model


def io_change(
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_onnx_file_path: Optional[str] = '',
    input_names: Optional[List[str]] = [],
    input_shapes: Optional[List[Union[int, str]]] = [],
    output_names: Optional[List[str]] = [],
    output_shapes: Optional[List[Union[int, str]]] = [],
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:
    """

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_onnx_file_path: Optional[str]
        Output onnx file path. If not specified, no ONNX file is output.\n\
        Default: ''

    input_names: Optional[List[str]]
        List of input OP names. All input OPs of the model must be specified.\n\
        The order is unspecified, but must match the order specified for input_shapes.\n\
        e.g. ['input.A', 'input.B', 'input.C']

    input_shapes: Optional[List[Union[int, str]]]
        List of input OP shapes. All input OPs of the model must be specified.\n\
        The order is unspecified, but must match the order specified for input_names.\n\
        e.g.\n\
        [\n\
            [1, 3, 'H', 'W'],\n\
            ['N', 3, 'H', 'W'],\n\
            ['-1', 3, 480, 640],\n\
        ]

    output_names: Optional[List[str]]
        List of output OP names. All output OPs of the model must be specified.\n\
        The order is unspecified, but must match the order specified for output_shapes.\n\
        e.g. ['output.a', 'output.b', 'output.c']

    output_shapes: Optional[List[Union[int, str]]]
        List of input OP shapes. All output OPs of the model must be specified.\n\
        The order is unspecified, but must match the order specified for output_shapes.\n\
        e.g.\n\
        [\n\
            [1, 3, 'H', 'W'],\n\
            ['N', 3, 'H', 'W'],\n\
            ['-1', 3, 480, 640],\n\
        ]

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False

    Returns
    -------
    io_changed_graph: onnx.ModelProto
        onnx ModelProto with modified INPUT and OUTPUT shapes.
    """

    # Unspecified check for input_onnx_file_path and onnx_graph
    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    # Other check
    if input_names is None or len(input_names) == 0:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'At least one input_names must be specified.'
        )
        sys.exit(1)

    if input_shapes is None or len(input_shapes) == 0:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'At least one input_shapes must be specified.'
        )
        sys.exit(1)

    if len(input_names) != len(input_shapes):
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The number of input_names and input_shapes must match.'
        )
        sys.exit(1)

    if output_names is None or len(output_names) == 0:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'At least one output_names must be specified.'
        )
        sys.exit(1)

    if output_shapes is None or len(output_shapes) == 0:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'At least one output_shapes must be specified.'
        )
        sys.exit(1)

    if len(output_names) != len(output_shapes):
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The number of output_names and output_shapes must match.'
        )
        sys.exit(1)

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)

    input_dicts = {name:shape for (name, shape) in zip(input_names, input_shapes)}
    output_dicts = {name:shape for (name, shape) in zip(output_names, output_shapes)}

    updated_model = update_inputs_outputs_dims(
        model=onnx_graph,
        input_dims=input_dicts,
        output_dims=output_dicts,
    )

    # Shape Estimation
    io_changed_graph = None
    try:
        io_changed_graph = onnx.shape_inference.infer_shapes(updated_model)
    except:
        if not non_verbose:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} '+
                'The input shape of the next OP does not match the output shape. '+
                'Be sure to open the .onnx file to verify the certainty of the geometry.'
            )

    # Save
    if output_onnx_file_path:
        onnx.save(io_changed_graph, output_onnx_file_path)

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

    # Return
    return io_changed_graph


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-if',
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='INPUT ONNX file path'
    )
    parser.add_argument(
        '-of',
        '--output_onnx_file_path',
        type=str,
        required=True,
        help='OUTPUT ONNX file path'
    )
    parser.add_argument(
        '-i',
        '--input_names',
        type=str,
        action='append',
        required=True,
        help='\
            List of input OP names. All input OPs of the model must be specified.\
            The order is unspecified, but must match the order specified for input_shapes. \
            e.g.\
            --input_names "input.A" \
            --input_names "input.B" \
            --input_names "input.C"'
    )
    parser.add_argument(
        '-is',
        '--input_shapes',
        type=str,
        nargs='+',
        action='append',
        required=True,
        help='\
            List of input OP shapes. All input OPs of the model must be specified. \
            The order is unspecified, but must match the order specified for input_names. \
            e.g. \
            --input_shapes 1 3 "H" "W" \
            --input_shapes "N" 3 "H" "W" \
            --input_shapes "-1" 3 480 640'
    )
    parser.add_argument(
        '-o',
        '--output_names',
        type=str,
        action='append',
        required=True,
        help='\
            List of output OP names. All output OPs of the model must be specified. \
            The order is unspecified, but must match the order specified for output_shapes. \
            e.g. \
            --output_names "output.a" \
            --output_names "output.b" \
            --output_names "output.c"'
    )
    parser.add_argument(
        '-os',
        '--output_shapes',
        type=str,
        nargs='+',
        action='append',
        required=True,
        help='\
            List of input OP shapes. All output OPs of the model must be specified. \
            The order is unspecified, but must match the order specified for output_shapes. \
            e.g. \
            --output_shapes 1 3 "H" "W" \
            --output_shapes "N", 3, "H", "W" \
            --output_shapes "-1" 3 480 640'
    )
    parser.add_argument(
        '-n',
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    input_onnx_file_path = args.input_onnx_file_path
    output_onnx_file_path = args.output_onnx_file_path
    input_names = args.input_names
    output_names = args.output_names
    non_verbose = args.non_verbose

    input_shapes = []
    for src in args.input_shapes:
        input_shape = []
        for s in src:
            try:
                val = literal_eval(s)
                if isinstance(val, int) and val >= 0:
                    input_shape.append(val)
                else:
                    input_shape.append(s)
            except:
                input_shape.append(s)
        input_shapes.append(input_shape)

    output_shapes = []
    for src in args.output_shapes:
        output_shape = []
        for s in src:
            try:
                val = literal_eval(s)
                if isinstance(val, int) and val >= 0:
                    output_shape.append(val)
                else:
                    output_shape.append(s)
            except:
                output_shape.append(s)
        output_shapes.append(output_shape)

    input_name_list = [name for name in input_names]
    input_shape_list = [name for name in input_shapes]
    output_name_list = [name for name in output_names]
    output_shape_list = [name for name in output_shapes]

    if not output_onnx_file_path:
        output_onnx_file_path = input_onnx_file_path

    # Load
    onnx_graph = onnx.load(input_onnx_file_path)

    # change
    io_changed_graph = io_change(
        input_onnx_file_path=None,
        onnx_graph=onnx_graph,
        output_onnx_file_path=output_onnx_file_path,
        input_names=input_name_list,
        input_shapes=input_shape_list,
        output_names=output_name_list,
        output_shapes=output_shape_list,
        non_verbose=non_verbose,
    )


if __name__ == '__main__':
    main()