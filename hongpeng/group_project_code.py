import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,7"
import warnings 
warnings.filterwarnings('ignore') 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
set_seed(42)
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json
import re


###### prediction model
def load_model(model_file):
    clf = pickle.load(open(model_file,"rb"))
    return clf

def prediction_model(jsonData):

    #load model
    clf = load_model('model.sav')


    #assign variables
    highBP = jsonData["highBP"]
    highChol = jsonData["highChol"]
    bmi = jsonData["bmi"]
    age = jsonData["age"]
    genHealth = jsonData["genHlth"]
    hrtDiseaseOrAttack = jsonData["heartDiseaseorAttack"]
    diffWalk = jsonData["diffWalk"]
   
    #organize in an array
    features = np.array([highBP, highChol, bmi, hrtDiseaseOrAttack, genHealth, diffWalk, age])
    prediction = clf.predict(features.reshape(1,-1))
    prob = clf.predict_proba(features.reshape(1,-1))
    return prediction, prob


###### language model
MODEL_NAME = "HuggingFaceH4/zephyr-7b-alpha"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, load_in_8bit=False, 
                                             device_map="auto", cache_dir="/home/scratch-buffalo/hjin008/huggingface_llm")
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
tokenizer.sep_token = tokenizer.unk_token
tokenizer.cls_token = tokenizer.unk_token
tokenizer.mask_token = tokenizer.unk_token
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto",)

def text2json_agent(txt):
    messages = [{"role": "system","content": "Your are a smart junior programmer."},
                {"role": "user", "content": f'''Please summarize the following text related personal information as a json.
                the json which only includes name, age, BMI, GenHlth(general health, scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor), 
                HighBP(0 = no, 1 = yes), DiffWalk(0 = no, 1 = yes), HighChol(0 = no, 1 = yes), and HeartDiseaseorAttack(0 = no, 1 = yes): {txt}'''}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True)
    # print(outputs[0]["generated_text"])
    # print('--'*60)
    text = outputs[0]["generated_text"]
    start = text.find("{")
    end = text.find("}", start) + 1
    json_text = text[start:end]
    json_text = re.sub(r'(?<!\\")//.*', '', json_text).replace('\n',' ')
    json_data = json.loads(json_text)
    # print(json_text)
    return json_data


def generate_response(user_input):
    # add the user's reponse to the chat records
    messages.append({"role": "user", "content": user_input})

    # transfer messages list by the chat template for llm better understanding and generating
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # llm generating response
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    # get the llm response
    generated_text = outputs[0]["generated_text"]
    assistant_response = generated_text[len(prompt):].strip()

    # add the AI assistant's response to the chat records
    messages.append({"role": "assistant", "content": assistant_response})

    return assistant_response

# remove some non-literal part to make the chat records clean
def post_process(text):
    for icon in ['<|>','\n', "<|assistant|>"]:
        text = text.replace(icon, "").strip()
    return text



###### main code

# system information and initail chat recode
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a professional nurse.",
    },
    {
        "role": "system", 
        "content": '''The following is a conversation between a assistant an user. As a hospital nurse, the assistant is engaging in a conversation with the user to inquire about some information including user's name, age, BMI, GenHlth(general health), HighBP, DiffWalk, HighChol, and HeartDiseaseorAttack. 
The conversation should be conducted in a dialogue format, with each question asked individually. The interaction should be characterized by a friendly and patient demeanor. 
The conversation will begin with a greeting from assistant, followed by the questions. Rememble ask one question a time.
After all the questions have been asked, assistant will summarize the information gathered and confirm its accuracy with user. Ok, take a deep breath, and let's go!'''},
]

# runing the chat function
while True:
    user_input = input("user: ")
    if "that is right." in user_input.lower(): 
        print("assistant: OK, now I am going to send the result to doctor, and I will be back soon. Thank you for your patience.")
        break
    response = generate_response(user_input)
    response = post_process(response)
    print("assistant:", response)

# change the summary information to JSON
res = text2json_agent(messages[-1]['content'].replace('<|assistant|>',' ').replace('\n',' '))


load_model("model.sav")
prediction, prob = prediction_model(res)
if prediction == 0 and prob < 0.7:
    result_txt = f"The user don't have the diabetes current, and the result shows the possibility of having diabetes is only {prob*100}%."
else: 
    result_txt = "The user have the diabetes current, and need to go to hospital to check."

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a professional nurse.",
    },
    {
        "role": "system", 
        "content": '''After collected the user's data, the doctor has given the diagnosis to you. Now your get the user's health condition. {result_txt} 
                      Now, your need to let the user know the result, taking care of the user's feeling if the result is bad, and welcome any questions from user. Let's go to talk to the user.'''
    },
    {
        "role": "assistant", 
        "content": '''Hello, I am back. Thank you again for the patience.'''
    },
]

print("assistant: Hello, I am back. Thank you again for the patience.")
while True:
    user_input = input("user: ")
    if "I don have any question." in user_input.lower(): 
        print("assistant: OK, have a nice day!")
        break
    response = generate_response(user_input)
    response = post_process(response)
    print("assistant:", response)