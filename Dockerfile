FROM python:3.8.18-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install boto3
RUN pip install Flask-Cors==4.0.0
RUN pip install numpy==1.24.0
RUN pip install pillow==10.2.0
RUN pip install Flask==3.0.2
RUN pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cpu


# Extractor Prefix
ENV "extractor_prefix" "/extractor"

# <translate_endpoint>/<prefix>
ENV "translate_endpoint" "http://translate-service.default.svc.cluster.local/translate"

# RUN pip install --trusted-host pypi.pytho.org -r ./backend-translation/requirements.txt
CMD ["python", "Extractor.py", "-t", "True"]
