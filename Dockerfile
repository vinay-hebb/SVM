# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the application code
COPY SVM_app.py .
COPY all_xi_ne_0.pkl .
COPY some_xi_ne_0.pkl .
COPY README.md .

# Expose the port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
