#!/bin/bash
cd /home/garuda_karura/sw2.5_bot
source venv/bin/activate
exec uvicorn qa_api:app --host 0.0.0.0 --port 8000

