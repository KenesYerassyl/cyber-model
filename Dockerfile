FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime
WORKDIR /workspace

# Copy all files to the working directory
COPY . /workspace

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter (torch is already in the base image)
RUN pip install jupyter

# Expose port for Jupyter Notebook
EXPOSE 8888

# Optional: Command to run Jupyter Notebook (uncomment if needed)
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
CMD ["python", "dann_training.py"]
# docker run --name cyber_container --gpus all -d -v /home/gahmlab1/cyber:/workspace cyber_image
# docker run --name cyber_container --gpus all -d -v /home/gahmlab1/cyber:/workspace -p 8888:8888 cyber_image
# jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root