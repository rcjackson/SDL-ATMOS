import subprocess
import os
import yaml
import pandas as pd
import logging

from io import StringIO
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_job_file(directions, nodes, ip, username, password, south, filename="dynamic_scan_job.yaml"):
    """
    Creates a job file dynamically for the Waggle scheduler.

    Parameters:
        directions (list): List of directions for scanning (e.g., ["NEH", "NEB", "NEG"]).
        nodes (dict): Dictionary of node names and their statuses (e.g., {"W020": True}).
        ip (str): IP address of the camera (e.g., "camera-mobotix-thermal").
        username (str): Username for the camera (e.g., "admin").
        password (str): Password for the camera (e.g., "wagglesage").
        south (str): South parameter value (e.g., "22").
        filename (str): The name of the output YAML file.

    Returns:
        str: The name of the generated job file.
    """
    job = {
        "name": "mobotix-scan-direction",
        "plugins": [
            {
                "name": "mobotix-scan-direction",
                "pluginSpec": {
                    "image": "registry.sagecontinuum.org/bhupendraraut/mobotix-scan:0.24.8.20",
                    "args": [
                        "--ip",
                        ip,
                        "--mode",
                        "direction",
                        "-south",
                        south,
                        "-pt",
                        f"{','.join(directions)}",
                        "-u",
                        username,
                        "-p",
                        password
                    ]
                }
            }
        ],
        "nodes": nodes,
        "scienceRules": [
            'schedule("mobotix-scan-direction"): cronjob("mobotix-scan-direction", "* * * * *")'
        ],
        "successCriteria": []
    }

    with open(filename, "w") as file:
        yaml.dump(job, file, default_flow_style=False)
    print(f"Job file {filename} created successfully.")
    return filename

def submit_job(filename):
    try:
        logging.info("Setting SES environment variables.")
        subprocess.run(["export", "SES_HOST=https://es.sagecontinuum.org"], shell=True)
        
        logging.info("Fetching SES_USER_TOKEN from environment.")
        token = os.environ.get('SES_USER_TOKEN')
        if not token:
            raise ValueError("API token not found in environment")
        else:
            logging.info("API token found.")

        logging.info("Is there already a mobotix-scan-direction job?")
        result = subprocess.run(["./sesctl", "stat"], check=True, capture_output=True,
                text=True)
        job_dataframe = pd.read_csv(StringIO(result.stdout), delim_whitespace=True)
        job_name = job_dataframe.where(job_dataframe["NAME"] == "mobotix-scan-direction").dropna()
        job_name = job_dataframe.where(job_name["STATUS"] == "Running").dropna()
        print(job_name)
        if len(job_name["NAME"].values) == 0:
            result = subprocess.run(["./sesctl", "create", "--file-path", filename], check=True, capture_output=True, text=True)
            logging.info("Creating job.")
            logging.info(f"Job creation response: {result.stdout}")
        
            logging.info("Extracting job_id from the response.")
            job_id = yaml.safe_load(result.stdout).get("job_id")
            if not job_id:
                raise ValueError("Job ID not found in the response.")
        else:
            job_id = str(int(job_name["JOB_ID"].values[0]))
            result = subprocess.run(["./sesctl", "edit", job_id, "--file-path", filename], check=True, capture_output=True, text=True)
            logging.info("Modifying job {job_id}.")
            logging.info(f"Job modified response: {result.stdout}")

        logging.info(f"Submitting job with job_id: {job_id}.")
        result = subprocess.run(["./sesctl", "submit", "--job-id", job_id], check=True, capture_output=True, text=True)
        logging.info(f"Job submission response: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during job submission: {e.stderr}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


