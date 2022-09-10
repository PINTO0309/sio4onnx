# sio4onnx
Simple tool to change the INPUT and OUTPUT shape of ONNX.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/sio4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sio4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/sio4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/sio4onnx?color=2BAF2B)](https://pypi.org/project/sio4onnx/) [![CodeQL](https://github.com/PINTO0309/sio4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/sio4onnx/actions?query=workflow%3ACodeQL)

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/178511831-14c1e72c-7816-47ea-9c13-27426066cd51.png" />
</p>

## 1. Setup
### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& pip install -U sio4onnx
```
### 1-2. Docker
https://github.com/PINTO0309/simple-onnx-processing-tools#docker

## 2. CLI Usage
```bash
$ sio4onnx -h

usage:
    sio4onnx [-h]
    -if INPUT_ONNX_FILE_PATH
    -of OUTPUT_ONNX_FILE_PATH
    -i INPUT_NAMES
    -is INPUT_SHAPES [INPUT_SHAPES ...]
    -o OUTPUT_NAMES
    -os OUTPUT_SHAPES [OUTPUT_SHAPES ...]
    [-n]

optional arguments:
  -h, --help
        Show this help message and exit.

  -if INPUT_ONNX_FILE_PATH, --input_onnx_file_path INPUT_ONNX_FILE_PATH
        INPUT ONNX file path

  -of OUTPUT_ONNX_FILE_PATH, --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
        OUTPUT ONNX file path

  -i INPUT_NAMES, --input_names INPUT_NAMES
        List of input OP names. All input OPs of the model must be specified.
        The order is unspecified, but must match the order specified for input_shapes.
        e.g.
        --input_names "input.A" \
        --input_names "input.B" \
        --input_names "input.C"

  -is INPUT_SHAPES [INPUT_SHAPES ...], --input_shapes INPUT_SHAPES [INPUT_SHAPES ...]
        List of input OP shapes. All input OPs of the model must be specified.
        The order is unspecified, but must match the order specified for input_names.
        e.g.
        --input_shapes 1 3 "H" "W" \
        --input_shapes "N" 3 "H" "W" \
        --input_shapes "-1" 3 480 640

  -o OUTPUT_NAMES, --output_names OUTPUT_NAMES
        List of output OP names. All output OPs of the model must be specified.
        The order is unspecified, but must match the order specified for output_shapes.
        e.g.
        --output_names "output.a" \
        --output_names "output.b" \
        --output_names "output.c"

  -os OUTPUT_SHAPES [OUTPUT_SHAPES ...], --output_shapes OUTPUT_SHAPES [OUTPUT_SHAPES ...]
        List of input OP shapes. All output OPs of the model must be specified.
        The order is unspecified, but must match the order specified for output_shapes.
        e.g.
        --output_shapes 1 3 "H" "W" \
        --output_shapes "N", 3, "H", "W" \
        --output_shapes "-1" 3 480 640

  -n, --non_verbose
        Do not show all information logs. Only error logs are displayed.
```

## 3. In-script Usage
```python
>>> from sio4onnx import io_change
>>> help(io_change)

Help on function io_change in module sio4onnx.onnx_input_output_variable_changer:

io_change(
    input_onnx_file_path: Union[str, NoneType] = '',
    onnx_graph: Union[onnx.onnx_ml_pb2.ModelProto, NoneType] = None,
    output_onnx_file_path: Union[str, NoneType] = '',
    input_names: Union[List[str], NoneType] = [],
    input_shapes: Union[List[Union[int, str]], NoneType] = [],
    output_names: Union[List[str], NoneType] = [],
    output_shapes: Union[List[Union[int, str]], NoneType] = [],
    non_verbose: Union[bool, NoneType] = False,
) -> onnx.onnx_ml_pb2.ModelProto

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.
        Either input_onnx_file_path or onnx_graph must be specified.
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.
        Either input_onnx_file_path or onnx_graph must be specified.
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_onnx_file_path: Optional[str]
        Output onnx file path. If not specified, no ONNX file is output.
        Default: ''

    input_names: Optional[List[str]]
        List of input OP names. All input OPs of the model must be specified.
        The order is unspecified, but must match the order specified for input_shapes.
        e.g. ['input.A', 'input.B', 'input.C']

    input_shapes: Optional[List[Union[int, str]]]
        List of input OP shapes. All input OPs of the model must be specified.
        The order is unspecified, but must match the order specified for input_names.
        e.g.
        [
            [1, 3, 'H', 'W'],
            ['N', 3, 'H', 'W'],
            ['-1', 3, 480, 640],
        ]

    output_names: Optional[List[str]]
        List of output OP names. All output OPs of the model must be specified.
        The order is unspecified, but must match the order specified for output_shapes.
        e.g. ['output.a', 'output.b', 'output.c']

    output_shapes: Optional[List[Union[int, str]]]
        List of input OP shapes. All output OPs of the model must be specified.
        The order is unspecified, but must match the order specified for output_shapes.
        e.g.
        [
            [1, 3, 'H', 'W'],
            ['N', 3, 'H', 'W'],
            ['-1', 3, 480, 640],
        ]

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.
        Default: False

    Returns
    -------
    io_changed_graph: onnx.ModelProto
        onnx ModelProto with modified INPUT and OUTPUT shapes.
```

## 4. CLI Execution
```bash
$ sio4onnx \
--input_onnx_file_path yolov3-10.onnx \
--output_onnx_file_path yolov3-10_upd.onnx \
--input_names "input_1" \
--input_names "image_shape" \
--input_shapes "batch" 3 "H" "W" \
--input_shapes "batch" 2 \
--output_names "yolonms_layer_1/ExpandDims_1:0" \
--output_names "yolonms_layer_1/ExpandDims_3:0" \
--output_names "yolonms_layer_1/concat_2:0" \
--output_shapes 1 "boxes" 4 \
--output_shapes 1 "classes" "boxes" \
--output_shapes "boxes" 3
```

## 5. In-script Execution
```python
from sio4onnx import io_change

io_changed_graph = io_change(
    input_onnx_file_path="yolov3-10.onnx",
    output_onnx_file_path="yolov3-10_upd.onnx",
    input_names=[
        "input_1",
        "image_shape",
    ],
    input_shapes=[
        ["batch", 3, "H", "W"],
        ["batch", 2],
    ],
    output_names=[
        "yolonms_layer_1/ExpandDims_1:0",
        "yolonms_layer_1/ExpandDims_3:0",
        "yolonms_layer_1/concat_2:0",
    ],
    output_shapes=[
        [1, "boxes", 4],
        [1, "classes", "boxes"],
        ["boxes", 3],
    ],
)
```
## 6. Sample
### Before
![image](https://user-images.githubusercontent.com/33194443/178515405-42d2bd01-f5fa-41be-95e3-3a229b0c8ae9.png)
### After
![image](https://user-images.githubusercontent.com/33194443/178515314-ecbf7f85-5c1d-4626-ac8b-3558432f6e9b.png)
