from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import mdtex2html
from gla.modeling_gla import GLAForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("/data/home/scyb066/modelbase/gla-chat-final", trust_remote_code=True, use_fast=False)
model = GLAForCausalLM.from_pretrained("/data/home/scyb066/modelbase/gla-chat-final", trust_remote_code=True).half().cuda()
model = model.eval()

# Override Chatbot.postprocess to render markdown responses
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess


# A helper function to parse and escape markdown content
def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    return "".join(lines)


# Function to handle user input and model inference
def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))
    
    # Prepare the messages for the model (applying chat template)
    messages = [
        {"role": "system", "content": "你是一个南湖医疗大模型，你可以回答关于医疗的问题。"},
        {"role": "user", "content": input}
    ]
    
    # Tokenize and format input
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    # Generate the response
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_length,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Process and decode the generated response
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Update chatbot history with input and response
    chatbot[-1] = (parse_text(input), parse_text(response))
    
    return chatbot, history


# Function to reset user input field
def reset_user_input():
    return gr.update(value='')


# Function to reset chatbot history
def reset_state():
    return [], []


# Build the Gradio interface
with gr.Blocks() as demo:
    gr.HTML("<h1 align='center'>百草汇智</h1>")

    chatbot = gr.Chatbot()
    
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(
                    show_label=False, 
                    placeholder="Input...", 
                    lines=10, 
                    container=False
                )
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    # Bind the submit button to the prediction function
    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    # Bind the reset button to clear chatbot history
    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

# Launch the Gradio app
demo.queue().launch(share=True, inbrowser=True)
