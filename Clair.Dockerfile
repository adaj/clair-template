# Use the specified base image
FROM base_clair:v0.4.0 AS build

# Set metadata
LABEL author="Adelson de Araujo (a.dearaujo@utwente.nl)"

# Set the working directory
WORKDIR /clair

# Create appuser and set permissions for clair directory
RUN useradd appuser && chown -R appuser /clair

# Copy your application files
COPY --chown=appuser:appuser ./clair /clair

# Switch to the appuser
USER appuser

# Expose necessary ports
EXPOSE 8000

# Set the default command to run uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/chatBot"]