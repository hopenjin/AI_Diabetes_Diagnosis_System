# AI_Diabetes_Diagnosis_System
<b> The group project of intro to ai</b>

<b>guidance</b> 
1. download or fork the repo to your local device.
2. install or update the Python package using the requirement_gpu_machine.txt and requirement_localhost.txt
3. move the "server" folder to your GPU machine, and make sure the machine has 30 Gib GPU ram
4. move the "local" folder to your localhost
5. use python to run the ai_nurse_llm.py on your GPU machine. This step needs some time to wait.
   python ./ai_nurse_llm.py
7. if the GPU machine and the localhost are the same device, it is fine. If not, please let the device with the "local" folder SSH connect to the port (default:5000) in your GPU machine.
   ssh -L 5000:localhost:5000 hjin008@onyx.cs.fiu.edu

9. 
10. On your localhost, open the terminal and move to the address of the "local" folder, then run the following code.
    uvicorn diabetes_server:app --reload
13. Open a new terminal and move to the address of the "local" folder, then run the following code.
    python -m http.server 8080
14. use a chrome browner ti open the link http://localhost:8080/
15. After all steps go well, you can talk with our AI nurse now.
