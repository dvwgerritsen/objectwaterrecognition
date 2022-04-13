FROM python:3.9-slim

COPY ./requirements.txt /requirements.txt

# This is required for PyTorch
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r /requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html && pip3 install gunicorn
WORKDIR /src
COPY ./ /src
CMD [ "gunicorn", "-b" , "0.0.0.0:5000", "app:app"]