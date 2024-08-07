import logging 
import sys
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
logging_dir = 'logs'
logfile_path = os.path.join(logging_dir, 'running_logs.logs')
os.makedirs(logging_dir, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(logfile_path),
        # logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("CollionLogger")
