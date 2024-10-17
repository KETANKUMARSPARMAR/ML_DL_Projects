import os
import pandas as pd
import mailparser
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


case_summary_tab = {
    "matter_overview": "This an investigation into alleged discrimination. The Enron corporation is a large, comodities, and services company based on Houston, Texas. This organization accused of hiding fraudlent details about its losses and debts from investors and creditors.",
    "people_aliases": "Kenneth Lay(CEO) \n Jeffrey Skilling (Former CEO and President) \n Andrew Fastow (Enron's CFO) \n Arthur Andersen LLP \n Sherron Watkins (Enron's Vice President of Corporate Development) \n Richard Causey (Chief Accounting Officer) \n Michael Kopper (Managing Director of Finance) \n David Ducan (Lead Auditor)",
    "noteworthy_orgs": "The Enron Corporation \n Arthur Andersen (Accounting Firm) \n Merrill Lynch \n Citigroup \n JPMorgan Chase"
}

prompt_criteria = """ System: I want you to act as an experience lawyer or attorney. Your task is to analyse entire document and identify or specify the document is relevance to given following informations only, do not use your own knowledge
<Legal_Matter>: I will provide you whole matter of the case. 
<People>: I will peovide you possible people or aliases who are involved in the legal case. 
<Noteworthy_Organizations>: I will provide you possible noteworthy organizations which are involved in the legal case.
<Document>: I will provide you the document to review.
Answer me as follow:
- Relevance: Is the document is relevant or not in one word only
- Reason: Explain why the given document is relevent or not in short
- Score: Give a score in the range of 1 to 10
<Legal_Matter>: {MATTER}
<People>: {PEOPLE}
<Noteworthy_Organizations>: {NOTEWORTHY_ORGS}
<Document>: {DOCUMENT}
Answer: """
#.format(MATTER = case_summary_tab['matter_overview'], PEOPLE = case_summary_tab['people_aliases'], NOTEWORTHY_ORGS = case_summary_tab['noteworthy_orgs'], DOCUMENT = "")

model = OllamaLLM(model="phi3.5", temperature=0)
prompt = ChatPromptTemplate.from_template(prompt_criteria)
chain = prompt | model

def start_ai_prediction():
    print("Welcome to AI!")
    nRowsRead = 2 # specify 'None' if want to read whole file
    # emails.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
    df1 = pd.read_csv('./data/emails.csv', delimiter=',', nrows = nRowsRead)
    df1.dataframeName = 'emails.csv'
    nRow, nCol = df1.shape
    #print(f'There are {nRow} rows and {nCol} columns')
    #print(df1.head(5))
    for index, row in df1.iterrows():
        msg = row['message']
        result = chain.invoke({"MATTER": case_summary_tab['matter_overview'], "PEOPLE": case_summary_tab['people_aliases'], "NOTEWORTHY_ORGS": case_summary_tab['noteworthy_orgs'], "DOCUMENT": msg})
        print("Document ID: ", index + 1)
        print(result)
        print("===========================================================================================================================\n")

if __name__=="__main__":
    start_ai_prediction()
