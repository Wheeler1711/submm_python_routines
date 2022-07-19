FROM jupyter/base-notebook
WORKDIR /submm
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
ENV PYTHONPATH="/submm"
