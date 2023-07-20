FROM python:3.8

WORKDIR /app

RUN pip install --upgrade pip

ADD . .

RUN pip install -e .

ENTRYPOINT [ "python", "src/deep.py" ]