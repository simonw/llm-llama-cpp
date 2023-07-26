# llm-llama-cpp

[![PyPI](https://img.shields.io/pypi/v/llm-llama-cpp.svg)](https://pypi.org/project/llm-llama-cpp/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-llama-cpp?include_prereleases&label=changelog)](https://github.com/simonw/llm-llama-cpp/releases)
[![Tests](https://github.com/simonw/llm-llama-cpp/workflows/Test/badge.svg)](https://github.com/simonw/llm-llama-cpp/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-llama-cpp/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin for running models using llama.cpp

## Installation

Install this plugin in the same environment as `llm`.
```bash
llm install llm-llama-cpp
```
## Adding models

After installation you will need to add or download some models.

This tool should work with any model that works with `llama.cpp`.

The plugin can download models for you. Try running this command:

```bash
llm llama-cpp download-model \
  https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin \
  --alias l2chat
```
This will download the Llama 2 7B Chat GGML model file (this one is 6.67GB), save it and register it with the plugin.

If you have already downloaded a `llama.cpp` compatible model you can tell the plugin to read it from its current location like this:

```bash
llm llama-cpp add-models path/to/model.bin
```
The model filename (minus the `.bin` extension) will be registered as its ID for executing the model. You can also set one or more aliases with both the `--alias` option.

## Running a prompt through a model

Once you have downloaded and added a model, you can run a prompt like this:
```bash
llm -m llama-2-7b-chat.ggmlv3.q8_0 'five names for a cute pet skunk'
```
Or if you registered an alias you can use that instead:
```bash
llm -m l2chat 'five creative names for a pet hedgehog'
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
