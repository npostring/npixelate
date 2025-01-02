FROM python:3.10-slim
WORKDIR /app
RUN apt update && apt upgrade -y 
RUN apt install libopencv-dev -y
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "view.py", "--server.port=8501", "--server.address=0.0.0.0"]