# Use the official Python 3.8 slim-buster image as the base image
# "slim-buster" is a lightweight version of the Debian-based image, suitable for production
FROM python:3.8-slim-buster

# Update the package list and install AWS CLI
# The '-y' flag automatically answers 'yes' to prompts and runs the installation non-interactively
RUN apt update -y && apt install awscli -y

# Set the working directory inside the container to /app
# All subsequent commands will be run in this directory
WORKDIR /app

# Copy the contents of the current directory on the host machine to the /app directory in the container
# This usually includes your application code and any other necessary files
COPY . /app

# Install the Python dependencies specified in the requirements.txt file
# This ensures all necessary packages are available for your application
RUN pip install -r requirements.txt

# Upgrade the 'accelerate' package to ensure you have the latest version
# This is useful for ensuring you have the most recent improvements and bug fixes
RUN pip install --upgrade accelerate

# Uninstall and then reinstall the 'transformers' and 'accelerate' packages
# This can be useful to ensure that the latest versions of these packages are installed, especially if there were issues with previous installations
RUN pip uninstall -y transformers accelerate
RUN pip install transformers accelerate

# Specify the command to run your application
# The container will execute 'python3 app.py' when it starts
CMD ["python3", "app.py"]
