# Read Me

## 1. To build docker image

```shell
docker build -t flask/peter-api . 
```

## 2. To run docker image

```shell
docker run -d -p 5000:5000 flask/peter-api  
```

## 3. To run docker image on port 80

```shell
docker run -d -p 80:5000 flask/peter-api  
```
