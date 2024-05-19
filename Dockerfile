FROM python:3.9

WORKDIR /code

EXPOSE 80

COPY . . 

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "streamlit", "run", "Home.py" ]