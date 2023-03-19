install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	docker run --rm -i hadolint/hadolint < Dockerfile
	pylint --disable=R,C,W1203,W0702 app.py

build:
	docker build -t flask-app:latest .

run:
	docker run -p 5000:5000 flask-app

invoke:
	curl http://127.0.0.1:5000

run-kube:
	kubectl apply -f kube-deploy.yaml

all: install lint test