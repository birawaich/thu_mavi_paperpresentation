# Paper Presentation: Depth Anything

Repository for a paper presentation of Depth Anything [V1](https://arxiv.org/abs/2401.10891) and [V2](https://arxiv.org/abs/2406.09414) for the Class _Machine Vision_ at [Tsinghua University](https://www.tsinghua.edu.cn/en/) in Fall 2024.

It mainly contains:

- Presentation (LaTex Beamer Source File) in `/presentation/`
- Python Demo at `run.py`

## Presentation

The presentation is a LaTex Beamer presentation, with the source file being `main.tex`. It compiles with `pdflatex`.

## Demo

The demonstration makes use of the [Huggingface Pipline](https://huggingface.co/docs/transformers/model_doc/depth_anything_v2) and offers live capturing from the webcam or converting single images. What is done and what pipeline is used can set by editing `run.py`.

If you are looking for a quick demo, check out the [Huggingface Live Demo](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2).

### Installation

1. Use a virtual environment with python 3.11, cf. [xkcd](https://xkcd.com/1987/). On Linux: `python3.11 -m venv venv` (creates a virtual enviornment in the current directory called `venv`)
2. Activate the environmnet and install the requirements. On Linux: `source venv/bin/activate` to activate and `pip install -r requirements.txt` to install the requirements.
3. Run the demonstration! On Linux: `python run.py`
