import os
import yaml
import llama
import inspect
import requests
import streamlit as st 
from llama import Type, Context
  
class Question(Type):
  function_name: str = Context("name of the function to describe")
  function_code: str = Context("text for the python code in the function")
  question_str: str = Context("question about the function")

class Answer(Type):
  parameter: list = Context("list of inputs to the function")
  output: list = Context("list of outputs from the function")
  description: str = Context("function description in 2 to 5 lines")

def create_model(name):
    model = llama.LLM(name=name)
    return model

def get_func_code(user_str):
    func_path = user_str.split('.')
    attr = llama
    for attr_string in func_path:
        attr = getattr(attr, attr_string)
    return inspect.getsource(attr)

def create_question(input_func_name):
    user_question = Question(
        function_name=input_func_name,
        function_code=get_func_code(input_func_name),
        question_str = f"Give API reference documentation for {input_func_name}"
        )
    return user_question

def get_function_docs(user_question, model):
    question = create_question(user_question)
    answer = model(input=question, output_type=Answer)
    docstring = user_question + "  \n  \n"
    docstring += "Parameters:  \n  \n"
    for parameter in answer.parameter:
        docstring += parameter + "  \n  \n" 
    docstring += "Output:  \n  \n" 
    for output in answer.output:
        docstring += output + "  \n  \n" 
    docstring += "Description:  \n  \n" + answer.description

    return docstring

model = create_model("code_analysis")

st.title("Code Understanding using llama-llm")

user_question = st.text_input(
    "Enter function name with path to explain : ",
    placeholder = "LLM.add_data",
)

if st.button("Tell me about it", type="primary"):
    try:
        function_docstring = get_function_docs(user_question, model)
        print(function_docstring)
        st.success(function_docstring)
    except:
        st.error("Error creating function doc, check that full function path is passed correctly, e.g. 'LLM.add_data'")