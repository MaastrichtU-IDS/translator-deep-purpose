# 🤿 Translator Deep Purpose

Using https://github.com/kexinhuang12345/DeepPurpose for the Translator project

## 🧶 Install

⚠️ Require python 3.8

You can use this docker image for development with Jupyterlab and VSCode integrated: `ghcr.io/maastrichtu-ids/jupyterlab:latest`

<details><summary>Create and activate virtual environment if necessary (no need in docker containers)</summary>

```bash
python -m venv .venv
source .venv/bin/activate
```

</details>

Install the required dependencies in the current environment:

```bash
pip install -e .
```

## Train

Run in the background, make sure the env variable `PYTHONUNBUFFERED` is set to 1 to get logs in `nohup.out`

```bash
nohup python src/train.py &
```

## 🛩️ Run

To get Deep Purpose predictions:

```bash
python src/deep.py
```