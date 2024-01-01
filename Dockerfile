FROM python:3.11
COPY . /src
WORKDIR /src
ADD main.py .
RUN pip install --upgrade pip
RUN pip install numpy==1.26.2 pandas==2.1.4 scikit-learn==1.3.2
CMD ["python", "main.py"]