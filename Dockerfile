FROM python:3.8
RUN mkdir /titanic-fairy
WORKDIR /titanic-fairy
COPY . .
ENV PYTHONPATH=${PYTHONPATH}:${PWD} 
RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install
RUN poetry shell 
RUN jupyter notebook 