FROM python:3.9.16-bullseye

# Loading all required files 
COPY requirements.txt ./requirements.txt
COPY train.csv ./train.csv
COPY test.csv ./test.csv

COPY train.py ./train.py
COPY app.py ./app.py

# Installing the dependencies 
RUN pip install -r requirements.txt

# Running the training
RUN python3 train.py

# Creating an entrypoint and pointing to a flask app file app.py
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]