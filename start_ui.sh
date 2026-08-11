#!/bin/bash
cd /home/garuda_karura/sw2.5_bot
source venv/bin/activate
exec streamlit run app.py --server.port 8501

