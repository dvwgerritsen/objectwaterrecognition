FROM python:3.8-slim

#RUN apt update && \
#    apt install --no-install-recommends -y build-essential gcc && \
#    apt clean && rm -rf /var/lib/apt/lists/*
COPY ./requirements.txt /requirements.txt
COPY ./src /src
RUN pip3 install --no-cache-dir -r /requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
EXPOSE 8080
WORKDIR /src
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]