Hugging Face's logo
Hugging Face
Search models, datasets, users...
Models
Datasets
Spaces
Posts
Docs
Solutions
Pricing



Spaces:

patpizio
/
llm-token-probs


like
0
App
Files
Community
llm-token-probs
/
app.py
patpizio's picture
patpizio
Update app.py
47fd2b6
4 months ago
raw
history
blame
contribute
delete
No virus
3.5 kB
import torch
import streamlit as st
import numpy as np
import plotly.express as px, plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, GenerationConfig, AutoModelForCausalLM

def top_token_ids(outputs, threshold=-np.inf):
    "Returns the index of the tokens whose score exceeds a threshold, for each output step"
    indexes = []
    for tensor in outputs['scores']:
        candidates = np.argwhere(tensor.flatten() > threshold).numpy()[0]
        ordering_mask = np.argsort(tensor[0][candidates])
        candidates = candidates[ordering_mask]
        if not isinstance(candidates, np.ndarray):
            indexes.append(np.array([candidates]))
        else:
            indexes.append(candidates)
    return indexes

def plot_word_scores(top_token_ids, outputs, tokenizer, boolq=False, width=600):
    fig = make_subplots(rows=len(top_token_ids), cols=1)
    for step, candidates in enumerate(top_token_ids):  
        fig.append_trace(
            go.Bar(
                y=[w[1:] for w in tokenizer.convert_ids_to_tokens(candidates)], 
                x=outputs['scores'][step][0][candidates], 
                orientation='h'
            ),
            row=step+1, col=1
        )
    fig.update_layout(
        width=500, 
        height=300*len(top_token_ids),
        showlegend=False
    )
    return fig

st.title('How do LLM choose their words?')

instruction = st.text_area(label='Write an instruction:', placeholder='Where is Venice located?')

col1, col2 = st.columns(2)

with col1:
    model_checkpoint = st.selectbox(
        "Model:",
        ("google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl")
    )

with col2:
    temperature = st.slider('Temperature:', min_value=0.0, max_value=1.0, value=0.5)
    top_p = st.slider('Top p:', min_value=0.5, max_value=1.0, value=0.99)
    # max_tokens = st.number_input('Max output length:', min_value=1, max_value=64, format='%i')
    max_tokens = st.slider('Max output length: ', min_value=1, max_value=64)
    # threshold = st.number_input('Min token score:: ', value=-10.0)
    
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = T5ForConditionalGeneration.from_pretrained(  
    model_checkpoint,
    load_in_8bit=False,
    device_map="auto",
    offload_folder="offload"
)


prompts = [
    f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction: {instruction}
    ### Response:"""
]

inputs = tokenizer(
    prompts[0],
    return_tensors="pt",
)
input_ids = inputs["input_ids"]#.to("cuda")

generation_config = GenerationConfig(
    do_sample=True,
    temperature=temperature,
    top_p=0.995,      # default 0.75
    top_k=100,        # default 80
    repetition_penalty=1.5,
    max_new_tokens=max_tokens,
)

if instruction:
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=generation_config,
            return_dict_in_generate=True, 
            output_scores=True
        )
    
    output_text = tokenizer.decode(
        outputs['sequences'][0],#.cuda(), 
        skip_special_tokens=False
    ).strip()
    
    st.write(output_text)

    fig = plot_word_scores(top_token_ids(outputs, threshold=-10.0), outputs, tokenizer)
    st.plotly_chart(fig, theme=None, use_container_width=False)
