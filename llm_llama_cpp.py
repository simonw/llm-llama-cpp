import click
import httpx
import io
import json
import llm
import os
import pathlib
import sys

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
    print(
        "llama_cpp not installed, install with: pip install llama-cpp-python",
        file=sys.stderr,
    )


def _ensure_models_dir():
    directory = llm.user_dir() / "llama-cpp" / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _ensure_models_file():
    directory = llm.user_dir() / "llama-cpp"
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / "models.json"
    if not filepath.exists():
        filepath.write_text("{}")
    return filepath


@llm.hookimpl
def register_models(register):
    directory = llm.user_dir() / "llama-cpp"
    models_file = directory / "models.json"
    if not models_file.exists():
        return
    models = json.loads(models_file.read_text())
    for model_id, details in models.items():
        register(LlamaModel(model_id, details["path"]), aliases=details["aliases"])


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def llama_cpp():
        "Commands for registering llama.cpp GGML models with LLM"

    @llama_cpp.command()
    def models_file():
        "Display the path to the models.json file"
        directory = llm.user_dir() / "llama-cpp"
        directory.mkdir(parents=True, exist_ok=True)
        models_file = directory / "models.json"
        click.echo(models_file)

    @llama_cpp.command()
    def models_dir():
        "Display the path to the directory holding downloaded models"
        click.echo(_ensure_models_dir())

    @llama_cpp.command()
    @click.argument("url")
    def download_model(url):
        "Download and register a model from a URL"
        if not url.endswith(".bin"):
            raise click.BadParameter("URL must end with .bin")
        with httpx.stream("GET", url, follow_redirects=True) as response:
            # Total size in bytes.
            total_size = response.headers.get("content-length")

            filename = url.split("/")[-1]
            # Set download path
            download_path = filename

            with open(download_path, "wb") as fp:
                if total_size is not None:  # If Content-Length header is present
                    total_size = int(total_size)
                    with click.progressbar(
                        length=total_size,
                        label="Downloading {}".format(human_size(total_size)),
                    ) as bar:
                        for data in response.iter_bytes(1024):
                            fp.write(data)
                            bar.update(len(data))
                else:  # If Content-Length header is not present
                    for data in response.iter_bytes(1024):
                        fp.write(data)

    @llama_cpp.command()
    @click.argument(
        "filepath", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
    )
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def add_model(filepath, aliases):
        "Register a GGML model you have already downloaded with LLM"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        path = pathlib.Path(filepath)
        model_id = path.stem
        models[model_id] = {
            "path": str(path.resolve()),
            "aliases": aliases,
        }
        models_file.write_text(json.dumps(models, indent=2))

    @llama_cpp.command()
    def models():
        "List registered GGML models"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        click.echo(json.dumps(models, indent=2))


class LlamaModel(llm.Model):
    class Options(llm.Options):
        verbose: bool = False

    def __init__(self, model_id, path):
        self.model_id = model_id
        self.path = path

    def execute(self, prompt, stream, response, conversation):
        with SuppressOutput(verbose=prompt.options.verbose):
            llm_model = Llama(model_path=self.path, verbose=prompt.options.verbose)
            stream = llm_model(prompt.prompt, stream=True, max_tokens=4000)
            for item in stream:
                # Each item looks like this:
                # {'id': 'cmpl-00...', 'object': 'text_completion', 'created': .., 'model': '/path', 'choices': [
                #   {'text': '\n', 'index': 0, 'logprobs': None, 'finish_reason': None}
                # ]}
                yield item["choices"][0]["text"]


def human_size(num_bytes):
    """Return a human readable byte size."""
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if num_bytes < 1024.0:
            break
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} {unit}"


class SuppressOutput:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            return
        # Save a copy of the current file descriptors for stdout and stderr
        self.stdout_fd = os.dup(1)
        self.stderr_fd = os.dup(2)

        # Open a file to /dev/null
        self.devnull_fd = os.open(os.devnull, os.O_WRONLY)

        # Replace stdout and stderr with /dev/null
        os.dup2(self.devnull_fd, 1)
        os.dup2(self.devnull_fd, 2)

        # Writes to sys.stdout and sys.stderr should still work
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = os.fdopen(self.stdout_fd, "w")
        sys.stderr = os.fdopen(self.stderr_fd, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            return
        # Restore stdout and stderr to their original state
        os.dup2(self.stdout_fd, 1)
        os.dup2(self.stderr_fd, 2)

        # Close the saved copies of the original stdout and stderr file descriptors
        os.close(self.stdout_fd)
        os.close(self.stderr_fd)

        # Close the file descriptor for /dev/null
        os.close(self.devnull_fd)

        # Restore sys.stdout and sys.stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
