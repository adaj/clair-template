# Streamlit.Dockerfile

# This file is used to build the Docker image for the Streamlit auxiliary apps

# Use the specified base image
FROM base_clair:v0.4.0

# Set metadata
LABEL author="Adelson de Araujo (a.dearaujo@utwente.nl)"

# Set the working directory
WORKDIR /clair

# Create appuser and set permissions for clair directory
RUN useradd appuser && chown -R appuser /clair

# Copy necessary application files for the Streamlit app
COPY --chown=appuser:appuser ./clair/triggering /clair/triggering
COPY --chown=appuser:appuser ./clair/app /clair/app
COPY --chown=appuser:appuser ./clair/nlu/intents /clair/nlu/intents

# Install Streamlit
RUN pip install streamlit

# Expose necessary port for Streamlit
EXPOSE 8501 8502 8503

# Set the default command to run streamlit
CMD ["bash", "app/streamlit.sh"]