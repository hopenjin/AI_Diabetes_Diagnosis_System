from flask import Flask, jsonify, request
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
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
    try:
        genHealth = jsonData['genHealth']
    except:
        genHealth = jsonData['GenHlth']
    hrtDiseaseOrAttack = jsonData["heartDiseaseorAttack"]
    diffWalk = jsonData["diffWalk"]
   
    #organize in an array
    features = np.array([highBP, highChol, bmi, hrtDiseaseOrAttack, genHealth, diffWalk, age])
    prediction = clf.predict(features.reshape(1,-1))
    prob = clf.predict_proba(features.reshape(1,-1))
    return prediction, prob

###### language model
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
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
                {"role": "system", "content": f"Please summarize the following text related personal information as a json, and the result value should be numbers. The json only includes age, BMI, genHealth(general health, scale 1-5, 1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor), HighBP(0 = no or false, 1 = yes or true), DiffWalk(0 = no or false, 1 = yes or true), HighChol(0 = no or false, 1 = yes or true), and HeartDiseaseorAttack(0 = no or false, 1 = yes or true): {txt}"}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=200, do_sample=True)
    text = outputs[0]["generated_text"]
    start = text.find("{")
    end = text.find("}", start) + 1
    json_text = text[start:end]
    json_text = re.sub(r'(?<!\\")//.*', '', json_text).replace('\n',' ')
    json_data = json.loads(json_text)
    return json_data


def generate_response(user_input, messages):
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
    for icon in ['<|>','\n', "<|assistant|>", '<|user|>']:
        text = text.replace(icon, "").strip()
    return text


messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a professional nurse, and the user comes here is for checking diabetes. You just record some important information from the conversation with the user.",
    },
    {
        "role": "system", 
        "content": '''The following is a conversation between a assistant an user. As a hospital nurse, you, the assistant, is engaging in a conversation with the user to inquire about some information, including user's name, age, BMI(just ask directly), GenHlth(general health score 1 to 5, 1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor), HighBP, DiffWalk, HighChol, and HeartDiseaseorAttack. 
The conversation should be conducted in a dialogue format, with each question asked individually. The interaction should be characterized by a friendly and patient demeanor. 
The conversation will begin with a greeting from assistant, followed by the questions. Rememble ask one question a time.
After all the questions have been asked, assistant will summarize the information gathered and confirm its accuracy with user. Ok, remember ask one question by one question.'''},
]

def language_model_response(user_input):
    if "that is right." in user_input.lower():
        response = generate_response(user_input, messages)
        for i in range(10):
            try: 

                res = text2json_agent(messages[-1]['content'].replace('<|assistant|>',' ').replace('\n',' '))
                load_model("model.sav")
                prediction, prob = prediction_model(res)
                prob = prob[0][1] 
                if prob < 0.7: 
                    prob = "{:.2f}%".format(prob * 100) 
                    result_txt = f"The user doesn't have the diabetes, and the result shows the possibility of having diabetes is only {prob}."
                else: 
                    prob = "{:.2f}%".format(prob * 100) 
                    result_txt = f"This user most likely has diabetes and need to go to hospital to check soon."
                    
                messages.append( {"role": "system", "content": f"The final diagnosis is that {result_txt}"})
                messages.append( {"role": "system", "content": f'''After collected the user's data, the doctor has given the diagnosis to you. Now your get the user's health condition. Now, your need to let the user know the result related to diabetes, taking care of the user's feeling if the result is bad, and welcome any questions from user. Let's go to talk to the user.'''})
                break
            except: 
                None

        # wait 3sec   
        import time
        time.sleep(3)
        
        messages.append( {"role": "assistant", "content": "OK, I got the result now."})
        return "OK, I got the result now."
        
    if "that is right." not in str(messages): 
        
        response = generate_response(user_input, messages)
        response = post_process(response)
        return response
        if "that is right." in long_string.lower():
            return "OK, now I am going to send the result to doctor, and I will be back soon. Thank you for your patience."
    else:
        
        if "I don have any question." in user_input.lower(): 
            return "OK, have a nice day!"
        else:
            response = generate_response(user_input, messages)
            response = post_process(response)    
        return response
    

###### Flask 
app = Flask(__name__)    
    
@app.route('/process-string', methods=['POST'])
def process_string():
    data = request.json
    long_string = data['text']
    length_of_string = len(long_string)
    result = f'length of this string is: {length_of_string}'
    return jsonify({'result': result})


@app.route('/language-model', methods=['POST'])
def language_model():
    data = request.json
    txt = data['text']
    if txt=='reset':
        messages = [
                    {
                        "role": "system",
                        "content": "You are a friendly chatbot who always responds in the style of a professional nurse, and the user comes here is for checking diabetes. You just record some important information from the conversation with the user.",
                    },
                    {
                        "role": "system", 
                        "content": '''The following is a conversation between a assistant an user. As a hospital nurse, you, the assistant, is engaging in a conversation with the user to inquire about some information, including user's name, age, BMI(just ask directly), GenHlth(general health score 1 to 5, 1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor), HighBP, DiffWalk, HighChol, and HeartDiseaseorAttack. 
                The conversation should be conducted in a dialogue format, with each question asked individually. The interaction should be characterized by a friendly and patient demeanor. 
                The conversation will begin with a greeting from assistant, followed by the questions. Rememble ask one question a time.
                After all the questions have been asked, assistant will summarize the information gathered and confirm its accuracy with user. Ok, remember ask one question by one question.'''},
                ]

        return jsonify({'text': 'reset done'})
    else:
        result = language_model_response(txt)
        return jsonify({'text': result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
