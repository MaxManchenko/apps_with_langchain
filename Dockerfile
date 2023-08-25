FROM python:3.8
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the app directory
COPY ./app /code/app

# Copy the configs directory
COPY ./configs /code/configs

# Copy the utils directory
COPY ./src/utils /code/src/utils

# Copy the model
COPY ./models /code/models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
