# llm-llama-cpp

[![PyPI](https://img.shields.io/pypi/v/llm-llama-cpp.svg)](https://pypi.org/project/llm-llama-cpp/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-llama-cpp?include_prereleases&label=changelog)](https://github.com/simonw/llm-llama-cpp/releases)
[![Tests](https://github.com/simonw/llm-llama-cpp/workflows/Test/badge.svg)](https://github.com/simonw/llm-llama-cpp/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-llama-cpp/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin for running models using [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Installation

Install this plugin in the same environment as `llm`.
```bash
llm install llm-llama-cpp
```
The plugin has an additional dependency on [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) which needs to be installed separately.

If you have a C compiler available on your system you can install that like so:
```bash
llm install llama-cpp-python
```
If you are using Python 3.11 installed via Homebrew on an M1 or M2 Mac you may be able to install this wheel instead, which will install a lot faster as it will not need to run a C compiler:
```bash
llm install https://static.simonwillison.net/static/2023/llama_cpp_python-0.1.77-cp311-cp311-macosx_13_0_arm64.whl
```
## Adding models

After installation you will need to add or download some models.

This tool should work with any model that works with `llama.cpp`.

The plugin can download models for you. Try running this command:

```bash
llm llama-cpp download-model \
  https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin \
  --alias llama2-chat --alias l2c
```
This will download the Llama 2 7B Chat GGML model file (this one is 6.67GB), save it and register it with the plugin - with two aliases, `llama2-chat` and `l2c`.

If you have already downloaded a `llama.cpp` compatible model you can tell the plugin to read it from its current location like this:

```bash
llm llama-cpp add-model path/to/llama-2-7b-chat.ggmlv3.q8_0.bin \
  --alias l27c
```
The model filename (minus the `.bin` extension) will be registered as its ID for executing the model.

You can also set one or more aliases using the `--alias` option.

You can see a list of models you have registered in this way like this:
```bash
llm llama-cpp models
```
Models are registered in a `models.json` file. You can find the path to that file in order to edit it directly like so:
```bash
llm llama-cpp models-file
```
For example, to edit that file in Vim:
```bash
vim "$(llm llama-cpp models-file)"
```
To find the directory with downloaded models, run:
```bash
llm llama-cpp models-dir
```
Here's how to change to that directory:
```bash
cd "$(llm llama-cpp models-dir)"
```

## Running a prompt through a model

Once you have downloaded and added a model, you can run a prompt like this:
```bash
llm -m llama-2-7b-chat.ggmlv3.q8_0 'five names for a cute pet skunk'
```
Or if you registered an alias you can use that instead:
```bash
llm -m llama2-chat 'five creative names for a pet hedgehog'
```

## More models to try

This model is Llama 2 7B GGML without the chat training. You'll need to prompt it slightly differently:
```bash
llm llama-cpp download-model \
  https://huggingface.co/TheBloke/Llama-2-7B-GGML/resolve/main/llama-2-7b.ggmlv3.q8_0.bin \
  --alias llama2
```
Try prompts that expect to be completed by the model, for example:
```bash
llm -m llama2 'Three fancy names for a posh albatross are:'
```


## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-llama-cpp
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
pytest
```
