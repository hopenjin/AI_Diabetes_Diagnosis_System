# AI Diabetes Diagnosis System
This repo is the complete code implementation of our paper, AI-Powered Conversational Diabetes Diagnosis System.

## Repo introduction
The repo includes the original code and the final code completion of this project. The users only need the "local" file and "server" file for the system deployment.

## How to use the repo?
1. Download or fork the repo to your local device.
2. Install or update the Python package using the requirement_gpu_machine.txt and requirement_localhost.txt. The Python version we used is 3.9.
```bash
   pip install requirement_gpu_machine.txt
```
```bash
   pip install requirement_localhost.txt
```
4. Move the "server" folder to your GPU machine, and make sure the machine has 30 Gib GPU RAM.
5. Move the "local" folder to your localhost
6. Use Python to run the ai_nurse_llm.py on your GPU machine. This step needs some time to wait.
   python ./ai_nurse_llm.py
7. If the GPU machine and the localhost are the same device, it is fine. If not, please let the device with the "local" folder SSH connect to the port (default:5000) in your GPU machine.
```bash
   ssh -L 5000:localhost:5000 "your GPU machine account and address"
```

8. On your localhost, open the terminal and move to the address of the "local" folder, then run the following code.

```bash
    uvicorn diabetes_server:app --reload
```
9. Open a new terminal and move to the address of the "local" folder, then run the following code.
```bash
    python -m http.server 8080
```
10. Use a Chrome browser to open the link http://localhost:8080/
11. After all steps go well, you can talk with our AI nurse now.
