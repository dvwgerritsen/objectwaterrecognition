# water-object-recognition
## Deployment

### Met Docker

De webapplicatie kan worden gestart in een Docker container.
Om de image te maken
```sudo docker build water-object-recognition -t object-recognition```

Om de image te starten
```sudo docker run -p 5000:5000 object-recognition```

Als de container is gestart is de webapplicatie te zien op http://127.0.0.1:5000

### Zonder Docker

Installeer de requirements
```pip3 install -r /requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html```

Start de applicatie
```python -m flask run```

Als de applicatie gestart kun je het openen op http://127.0.0.1:5000/ 
