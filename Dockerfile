FROM python:3.8
RUN mkdir /titanic-fairy
WORKDIR /titanic-fairy
COPY . .
ENV PYTHONPATH=${PYTHONPATH}:${PWD} 
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi
#To-Do run tests