#!/bin/bash

# Start app1 on port 8501
streamlit run app/auxiliary_apps/similarity.py --server.port 8501 &

# Start app2 on port 8502
streamlit run app/auxiliary_apps/blacklist.py --server.port 8502 &

# Start app3 on port 8503
streamlit run app/auxiliary_apps/intents.py --server.port 8503 &

wait
