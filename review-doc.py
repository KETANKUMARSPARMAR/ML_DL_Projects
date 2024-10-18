import os
import pandas as pd
import mailparser
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser



case_summary_tab = {
    "matter_overview": "This is an investigation into alleged discriminatory practices. Enron Corporation, a large commodities and services company based in Houston, Texas, is accused of concealing fraudulent information about its losses and debts from investors and creditors.", # "This an investigation into alleged discrimination. The Enron corporation is a large, comodities, and services company based on Houston, Texas. This organization accused of hiding fraudlent details about its losses and debts from investors and creditors.",
    "people_aliases": "Kenneth Lay - CEO (ken.lay@enron.com) \n Jeffrey Skilling - Former CEO and President (jeff.skilling@enron.com) \n Andrew Fastow - Enron's CFO (andrew.fastow@enron.com) \n Arthur Andersen LLP \n Sherron Watkins - Enron's Vice President of Corporate Development (sherron.watkins@enron.com) \n Richard Causey - Chief Accounting Officer (richard.causey@enron.com) \n Michael Kopper - Managing Director of Finance (michael.kopper@enron.com) \n David Ducan - Lead Auditor (david.duncan@arthurandersen.com )",
    "noteworthy_orgs": "The Enron Corporation \n Arthur Andersen (Accounting Firm) \n Merrill Lynch \n Citigroup \n JPMorgan Chase"
}

# prompt_criteria = """System: I want you to act as an experience lawyer or attorney. Your task is to analyse entire document and identify or specify the document is relevance to given following informations only, do not use your own knowledge
# <Legal_Matter>: I will provide you whole matter of the case. 
# <People>: I will peovide you possible people or aliases who are involved in the legal case. 
# <Noteworthy_Organizations>: I will provide you possible noteworthy organizations which are involved in the legal case.
# <Document>: I will provide you the document to review.
# Answer me as follow:
# - Relevance: Is the document is relevant or not in one word only
# - Reason: Explain why the given document is relevent or not in short
# - Score: Give a score in the range of 1 to 10
# <Legal_Matter>: {MATTER}
# <People>: {PEOPLE}
# <Noteworthy_Organizations>: {NOTEWORTHY_ORGS}
# <Document>: {DOCUMENT}
# Answer: """
#.format(MATTER = case_summary_tab['matter_overview'], PEOPLE = case_summary_tab['people_aliases'], NOTEWORTHY_ORGS = case_summary_tab['noteworthy_orgs'], DOCUMENT = "")

prompt_criteria = """System Prompt: You are an experienced lawyer. Your task is to analyze the provided document and determine its relevance based strictly on the following information. Do not use outside knowledge or make assumptions.
Inputs:
Legal Matter: The key legal issue or case details, which will be provided.
People Involved: A list of individuals or aliases relevant to the case.
Noteworthy Organizations: A list of organizations involved in the case.
Document: The document to be analyzed.

Instructions:
Based on the provided information, respond with the following:
Relevance: Answer with one word only ("Relevant" or "Not Relevant").
Reason: Provide a brief explanation of why the document is or is not relevant.
Score: Rate the relevance of the document on a scale from 1 to 10.

Input Format:
Legal Matter: {MATTER}
People Involved: {PEOPLE}
Noteworthy Organizations: {NOTEWORTHY_ORGS}
Document: {DOCUMENT}

Answer Format:
Relevance: [Relevant/Not Relevant]
Reason: [Brief explanation]
Score: [1-10]"""

substring_list = ["ken.lay@enron.com", "jeff.skilling@enron.com", "andrew.fastow@enron.com", "sherron.watkins@enron.com", "richard.causey@enron.com", "michael.kopper@enron.com", "Kenneth Lay", "Jeffrey Skilling", "Andrew S Fastow", "Richard Causey", "Sherron Watkins"]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0) # OllamaLLM(model="phi3.5", temperature=0)
prompt = ChatPromptTemplate.from_template(prompt_criteria)
chain = prompt | model | StrOutputParser()

def start_ai_prediction():
    print("Welcome to AI!")
    nRowsRead = 5000 # specify 'None' if want to read whole file
    # emails.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
    df1 = pd.read_csv('./data/emails.csv', delimiter=',', nrows = nRowsRead)
    df1.dataframeName = 'emails.csv'
    nRow, nCol = df1.shape
    #print(f'There are {nRow} rows and {nCol} columns')
    #print(df1.head(5))
    for index, row in df1.iterrows():
        msg = row['message']
        if any(map(msg.__contains__, substring_list)):
            result = chain.invoke({"MATTER": case_summary_tab['matter_overview'], "PEOPLE": case_summary_tab['people_aliases'], "NOTEWORTHY_ORGS": case_summary_tab['noteworthy_orgs'], "DOCUMENT": msg})
            #print(msg)
            with open("./review/result.txt", "a") as f:
                f.write("Document ID: {docID} - File: {file} \n -------+++Result+++------ \n{res} \n===========================================================================================================================\n\n".format(docID=index + 1, file=row['file'], res=result))

if __name__=="__main__":
    start_ai_prediction()