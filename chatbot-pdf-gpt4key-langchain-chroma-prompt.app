import gradio as gr
import os
import time
import pandas as pd
import sqlite3
import ocrmypdf

from langchain.document_loaders import OnlinePDFLoader #for laoding the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import RetrievalQA # for conversing with chatGPT
from langchain.chat_models import ChatOpenAI # the LLM model we'll use (ChatGPT)
from langchain import PromptTemplate

 
def load_pdf_and_generate_embeddings(pdf_doc, open_ai_key, relevant_pages):
    if open_ai_key is not None:
        os.environ['OPENAI_API_KEY'] = open_ai_key
        #OCR Conversion - skips conversion of pages that already contain text
        pdf_doc = ocr_converter(pdf_doc)
        #Load the pdf file
        loader = OnlinePDFLoader(pdf_doc)
        pages = loader.load_and_split()
        print('pages loaded:', len(pages))
        
        #Create an instance of OpenAIEmbeddings, which is responsible for generating embeddings for text
        embeddings = OpenAIEmbeddings()

        pages_to_be_loaded =[]

        if relevant_pages:
            page_numbers = relevant_pages.split(",")
            if len(page_numbers) != 0:
                for page_number in page_numbers:
                    if page_number.isdigit():
                        pageIndex = int(page_number)-1
                        if pageIndex >=0 and pageIndex <len(pages):
                            pages_to_be_loaded.append(pages[pageIndex])
        
        #In the scenario where none of the page numbers supplied exist in the PDF, we will revert to using the entire PDF.
        if len(pages_to_be_loaded) ==0:
            pages_to_be_loaded = pages.copy()
            
             
        #To create a vector store, we use the Chroma class, which takes the documents (pages in our case) and the embeddings instance
        vectordb = Chroma.from_documents(pages_to_be_loaded, embedding=embeddings)
        
        #Finally, we create the bot using the RetrievalQA class
        global pdf_qa

        #Configuring the Prompt Template is the key to getting the desired response in the desired format.
        prompt_template = """Use the following pieces of context to answer the question at the end. If you do not know the answer, just return N/A. If you encounter a date, return it in mm/dd/yyyy format. If there is a Preface section in the document, extract the chapter# and the short description from the Preface. Chapter numbers are listed to the left in Preface and always start with an alphabet, for example A1-1
        {context}
        Question: {question}
        Return the answer. Provide the answer in the JSON format and extract the key from the question. Where applicable, break the answer into bullet points. When the sentences are long, try and break them into sub sections and include all the information and do not skip any information. If there is an exception to the answer, please do include it in a 'Note:' section. If there are no exceptions to the answer, please skip the 'Note:' section. Include a 'For additional details refer to' section when the document has more information to offer on the topic being questioned. If the document has a Preface or 'Table of Contents' section, extract the chapter# and a short description and include the info under the 'For additional details refer to' section. List only the chapters that contain information or skip this section altogether. Do not use page numbers as chapter numbers as they are different. If additional information is found in multiple pages within the same chapter, list the chapter only once. If chapter information cannot be extracted, include any other information that will help the user navigate to the relevant sections of the document. If the document does not contain a Preface or 'Table of Contents' section, please do not call that out. For example, do not include statements like the following in the answer - 'The document does not contain a Preface or 'Table of Contents' section'""" 

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}
        pdf_qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0, model_name="gpt-4"),chain_type="stuff", retriever=vectordb.as_retriever(search_kwargs={"k": 5}), chain_type_kwargs=chain_type_kwargs, return_source_documents=False)
               
        return "Ready"
    else:
        return "Please provide an OpenAI gpt-4 API key"

def create_db_connection():
    DB_FILE = "./questionset.db"
    connection = sqlite3.connect(DB_FILE,check_same_thread=False)
    return connection

def create_sqlite_table(connection):
    print("*****Entered the create_sqlite_table method*****")
    cursor = connection.cursor()
    # Create table if it doesn't already exist
    try:
        data = f'SELECT * FROM questions'
        cursor.execute(data)
        cursor.fetchall()
    
    except sqlite3.OperationalError:
        cursor.execute(
        '''
        CREATE TABLE questions (document_type TEXT NOT NULL, questionset_tag TEXT NOT NULL, field TEXT NOT NULL, question TEXT NOT NULL)
        ''')
        print("*****questions table has been created******")
    connection.commit()

def load_master_questionset_into_sqlite(connection):
    create_sqlite_table(connection)
    cursor = connection.cursor()
    #Check to make sure the masterlist has not been loaded already.
    masterlist_for_DOT_count = cursor.execute("Select COUNT(document_type) from questions where document_type=? and questionset_tag=?",("DOT","masterlist",),).fetchone()[0]
    if masterlist_for_DOT_count == 0:
        print("DOT masterlist has not yet been loaded, proceeding to load.")
        #Create a list of questions around the relevant fields of a Deed of Trust(DOT) document
        fieldListForDOT, queryListForDOT = create_field_and_question_list_for_DOT()
        #Create a list of questions around the relevant fields of a TRANSMITTAL SUMMARY document
        fieldListForTransmittalSummary, queryListForTransmittalSummary = create_field_and_question_list_for_Transmittal_Summary()
        #Loop thru the list and add them into the questions table
        i = 0
        print("*****Entered the load master question set method*****")
        while i < len(queryListForDOT):
            cursor.execute("INSERT INTO questions(document_type, questionset_tag, field, question) VALUES(?,?,?,?)", ["DOT", "masterlist", fieldListForDOT[i], queryListForDOT[i]])
            i = i+1
        i = 0
        while i < len(queryListForTransmittalSummary):
            cursor.execute("INSERT INTO questions(document_type, questionset_tag, field, question) VALUES(?,?,?,?)", ["Transmittal Summary", "masterlist", fieldListForTransmittalSummary[i], queryListForTransmittalSummary[i]])
            i = i+1
        connection.commit()
    total_questions = cursor.execute("Select COUNT(document_type) from questions").fetchone()[0]
    print("*******Total number of questions in the DB:", total_questions)

def create_field_and_question_list_for_DOT():
    #Create a list of questions around the relevant fields of a Deed of Trust(DOT) document
    query1 = "what is the Loan Number?"
    field1 = "Loan Number"
    query2 = "Who is the Borrower?"
    field2 = "Borrower"
    query3 = "what is the Case Number?"
    field3 = "Case Number"
    query4 = "what is the Mortgage Identification number?"
    field4 = "MIN Number"
    query5 = "DOT signed date?"
    field5 = "Signed Date"
    query6 = "Who is the Lender?"
    field6 = "Lender"
    query7 = "what is the VA/FHA Number?"
    field7 = "VA/FHA Number"
    query8 = "Who is the Co-Borrower?"
    field8 = "Co-Borrower"
    query9 = "What is the property type - single family, multi family?"
    field9 = "Property Type"
    query10 = "what is the Property Address?"
    field10 = "Property Address"
    query11 = "In what County is the property located?"
    field11 = "Property County"
    query12 = "what is the Electronically recorded date"
    field12 = "Electronic Recording Date"
    queryList = [query1, query2, query3, query4, query5, query6, query7, query8, query9, query10, query11,query12]
    fieldList = [field1, field2, field3, field4, field5, field6, field7, field8, field9, field10, field11,field12]
    return fieldList, queryList

def create_field_and_question_list_for_Transmittal_Summary():
    #Create a list of questions around the relevant fields of a TRANSMITTAL SUMMARY document
    query1 = "Who is the Borrower?"
    field1 = "Borrower"
    query2 = "what is the Property Address?"
    field2 = "Property Address"
    query3 = "what is the Loan Term?"
    field3 = "Loan Term"
    query4 = "What is the Base Income?"
    field4 = "Base Income"
    query5 = "what is the Borrower's SSN?"
    field5 = "Borrower's SSN"
    query6 = "Who is the Co-Borrower?"
    field6 = "Co-Borrower"
    query7 = "What is the Original Loan Amount?"
    field7 = "Original Loan Amount"
    query8 = "What is the Initial P&I payment?"
    field8 = "Initial P&I payment"
    query9 = "What is the Co-Borrower's SSN?"
    field9 = "Co-Borrower’s SSN"
    query10 = "Number of units?"
    field10 = "Units#"
    query11 = "Who is the Seller?"
    field11 = "Seller"
    query12 = "Document signed date?"
    field12 = "Signed Date"
    queryList = [query1, query2, query3, query4, query5, query6, query7, query8, query9, query10, query11,query12]
    fieldList = [field1, field2, field3, field4, field5, field6, field7, field8, field9, field10, field11,field12]
    return fieldList, queryList

def retrieve_document_type_and_questionsettag_from_sqlite():
    connection = create_db_connection()
    load_master_questionset_into_sqlite(connection)
    cursor = connection.cursor()
    rows = cursor.execute("SELECT document_type, questionset_tag FROM questions order by document_type, upper(questionset_tag)").fetchall()
    print("Number of rows retrieved from DB:",len(rows))
    list_for_dropdown = []
    for i in rows:
      entries_in_row = list(i) 
      concatenated_value = entries_in_row[0]+ ":" + entries_in_row[1]
      if concatenated_value in list_for_dropdown:
          print("Value already in the list:", concatenated_value)
      else:
          list_for_dropdown.append(concatenated_value)
          print(concatenated_value)
    
    print("Number of unique entries found in the DB:",len(list_for_dropdown))
    connection.close()
    return gr.Dropdown.update(choices=list_for_dropdown,value=list_for_dropdown[0])


def retrieve_fields_and_questions(dropdownoption):
    #dropdownoption will be in the documentType:questionSetTag format
    print("dropdownoption is:", dropdownoption)
    splitwords = dropdownoption.split(":")
    connection = create_db_connection()
    cursor = connection.cursor()
    fields_and_questions = cursor.execute("SELECT document_type,field, question FROM questions where document_type=? and questionset_tag=?",(splitwords[0],splitwords[1],),).fetchall()
    connection.close()
    return pd.DataFrame(fields_and_questions, columns=["documentType","field", "question"])

def add_questionset(data, document_type, tag_for_questionset):
# loop through the rows using iterrows()
    connection = create_db_connection()
    create_sqlite_table(connection)
    cursor = connection.cursor()
    for index, row in data.iterrows():
        cursor.execute("INSERT INTO questions(document_type, questionset_tag, field, question) VALUES(?,?,?,?)", [document_type, tag_for_questionset, row['field'], row['question']])   
    connection.commit()
    connection.close()

def load_csv_and_store_questionset_into_sqlite(csv_file, document_type, tag_for_questionset):
    print('document type is:',document_type)
    print('tag_for_questionset is:',tag_for_questionset)
    
    if tag_for_questionset:
        if document_type:
            data = pd.read_csv(csv_file.name)
            add_questionset(data, document_type, tag_for_questionset)
            responseString = "Task Complete. Uploaded {} fields and the corresponding questions into the Database for {}:{}".format(data.shape[0], document_type,tag_for_questionset)
            return responseString
        else:
            return "Please select the Document Type and provide a name for the Question Set"
        

    
def answer_predefined_questions(document_type_and_questionset):
    print('chosen document_type_and_questionset:',document_type_and_questionset)
    option_chosen = document_type_and_questionset.split(":")
    document_type = option_chosen[0]
    question_set = option_chosen[1]
    fields =[]
    questions = []
    responses =[]
    connection = create_db_connection()
    cursor = connection.cursor()
    if document_type is not None:
        if question_set is not None:
            #Given the document_type and questionset_tag, retrieve the corresponding fields and questions from the database
            rows = cursor.execute("SELECT field, question FROM questions where document_type=? and questionset_tag=?",(document_type,question_set,),).fetchall()
            for i in rows:
              entries_in_row = list(i) 
              fields.append(entries_in_row[0])
              questions.append(entries_in_row[1])
              responses.append(pdf_qa.run(entries_in_row[1]))
        else:
            return "Please choose your Document Type:QuestionSet"

    return pd.DataFrame({"Field": fields, "Question to gpt-4": questions, "Response from gpt-4": responses})

    
def ocr_converter(input_file):
    image_pdf = input_file.name
    #ocrmypdf.ocr(image_pdf, image_pdf, skip_text=True, language="eng")
    ocrmypdf.ocr(image_pdf, image_pdf, redo_ocr=True, language="eng")
    return image_pdf

def summarize_contents():
    question = "Generate a short summary of the contents along with no more than 3 leading/example questions. Do not return the response in json format"
    return pdf_qa.run(question)    

def answer_query(query):
    question = query
    return pdf_qa.run(question)
    

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>AskMoli - Chatbot for PDFs</h1>
    <p style="text-align: center;">Upload a .PDF, click the "Upload PDF and generate embeddings" button, <br />
    Wait for the Status to show Ready. You can choose to get answers to the pre-defined question set OR ask your own question <br />
    The app is built on GPT-4 and leverages the magic of PromptTemplate</p>
</div>
"""

with gr.Blocks(css=css,theme=gr.themes.Monochrome()) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
    
    with gr.Tab("Chatbot"):
        with gr.Column():
            open_ai_key = gr.Textbox(label="Your GPT-4 OpenAI API key", type="password")
            pdf_doc = gr.File(label="Load a pdf",file_types=['.pdf'],type='file')
            relevant_pages = gr.Textbox(label="*Optional - List comma separated page numbers to load or leave this field blank to use the entire PDF")
        
            with gr.Row():
                status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf = gr.Button("Upload PDF and generate embeddings").style(full_width=False)

            with gr.Row():
                summary = gr.Textbox(label="Summary")
                summarize_pdf = gr.Button("Have Moli Summarize the Contents").style(full_width=False)

            with gr.Row():
                input = gr.Textbox(label="Type in your question")
                output = gr.Textbox(label="Answer")
                submit_query = gr.Button("Submit your own question to AskMoli").style(full_width=False)

            with gr.Row():
                questionsets = gr.Dropdown(label="Pre-defined Question Sets stored in the DB", choices=[])
                load_questionsets = gr.Button("Retrieve Pre-defined Question Sets from DB").style(full_width=False)
                fields_and_questions = gr.Dataframe(label="Fields and Questions in the chosen Question Set")
                load_fields_and_questions = gr.Button("Retrieve Pre-defined Questions from the DB for the chosen QuestionSet").style(full_width=False)
                            
            with gr.Row():
                answers = gr.Dataframe(label="Answers to Predefined Question set")
                answers_for_predefined_question_set = gr.Button("Get answers to the chosen pre-defined question set").style(full_width=False)
              
    with gr.Tab("OCR Converter"):
        with gr.Column():
            image_pdf = gr.File(label="Load the pdf to be converted",file_types=['.pdf'],type='file')
            
        with gr.Row():
            ocr_pdf =  gr.File(label="OCR'd pdf", file_types=['.pdf'],type='file',file_count="single")
            convert_into_ocr = gr.Button("Convert").style(full_width=False)

                
    with gr.Tab("Upload Question Set"):
        with gr.Column():
            document_types =["Mortgage 1040 US Individual Tax Returns 8453 Elec Form",
            "Mortgage 1098",
            "Mortgage 1099",
            "Mortgage Abstract",
            "Mortgage ACH Authorization Form",
            "Mortgage Advance Fee Agreement",
            "Mortgage Affidavit",
            "Mortgage Affidavit of Suspense Funds",
            "Mortgage Agreement Documents",
            "Mortgage Sales Contract",
            "Mortgage Loan Estimate",
            "Mortgage Alimony Or Child Support",
            "Mortgage Amended Proof Of Claim",
            "Mortgage Amortization Schedule",
            "Mortgage Flood Insurance",
            "Mortgage Appraisal Report",
            "Mortgage Appraisal Disclosure",
            "Mortgage ARM Letter",
            "Mortgage Arms Length Affidavit",
            "Mortgage Assignment-Recorded",
            "Mortgage Assignment-Unrecorded",
            "Mortgage Assignment of Rent or Lease",
            "Mortgage Automated Value Model",
            "Mortgage Award Letters",
            "Mortgage Bailee Letter",
            "Mortgage Balloon Disclosure",
            "Mortgage Bank Statement",
            "Mortgage Bankruptcy Documents",
            "Mortgage Bill of Sale",
            "Mortgage Billing Statement",
            "Mortgage Birth-Marriage-Death Certificate",
            "Mortgage Borrower Certification Authorization",
            "Mortgage Borrower Response Package",
            "Mortgage Brokers Price Opinion",
            "Mortgage Business Plan",
            "Mortgage Buydown Agreement",
            "Mortgage Bylaws Covenants Conditions Restrictions",
            "Mortgage Cash for Keys",
            "Mortgage Certificate of Redemption",
            "Mortgage Certificate of Sale",
            "Mortgage Certificate of Title",
            "Mortgage Certification of Amount Due Payoff Reinstatement",
            "Mortgage Checks-Regular or Cashiers",
            "Mortgage Closing Disclosure",
            "Mortgage Closing Protection Letter",
            "Mortgage Closing Other",
            "Mortgage Code Violations",
            "Mortgage Request for Release",
            "Mortgage Certificate of Liability Insurance",
            "Mortgage Commitment Letter",
            "Mortgage Complaint",
            "Mortgage Complaint Answer Counter Claim",
            "Mortgage Conditional Approval Letter",
            "Mortgage Conditional Commitment",
            "Mortgage Consent Order",
            "Mortgage Consolidated Mortgage CEMA",
            "Mortgage Conveyance Claims",
            "Mortgage Correction and Revision Agreement",
            "Mortgage Correspondence",
            "Mortgage Court Order Settlement Divorce Decree",
            "Mortgage Credit Report",
            "Mortgage Customer Signature Authorization",
            "Mortgage Debt Validation",
            "Mortgage Deed",
            "Mortgage Default Notices",
            "Mortgage Direct Debit Authorization Form",
            "Mortgage Disclosure Documents",
            "Mortgage Document Checklist",
            "Mortgage Document Correction and Fee Due Agreement",
            "Mortgage Dodd Frank Certification",
            "Mortgage Drivers License",
            "Mortgage Request for VOE",
            "Mortgage Environmental Indemnity Agreement",
            "Mortgage Equal Credit Opportunity Act Notice",
            "Mortgage Escrow Agreement",
            "Mortgage Escrow Analysis Trial Balance Worksheet",
            "Mortgage Instructions to Escrow Agent",
            "Mortgage Escrow Letters",
            "Mortgage Executed Deeds",
            "Mortgage Fair Lending Notice",
            "Mortgage Foreclosure Complaint",
            "Mortgage Foreclosure Judgement",
            "Mortgage Foreclosure Sale",
            "Mortgage FHA Neighborhood Watch",
            "Mortgage Truth-In-Lending Disclosure Statement",
            "Mortgage Financial Form",
            "Mortgage Financing Agreement",
            "Mortgage First Payment Letter",
            "Mortgage Forced Place Insurance Documents",
            "Mortgage Foreclosure Documents",
            "Mortgage Good Faith Estimate",
            "Mortgage Guaranty",
            "Mortgage HAMP Certifications",
            "Mortgage HOA-Condo Covenants and Dues",
            "Mortgage Exemption Hold Harmless Letter",
            "Mortgage Home Equity Signature Verification Card",
            "Mortgage Home Inspection",
            "Mortgage Property Liability Insurance",
            "Mortgage Homeowners Insurance Notice",
            "Mortgage HUD-1 Settlement Statement",
            "Mortgage Income Other",
            "Mortgage Indemnity Agreement",
            "Mortgage Informed Consumer Choice Disclosure Notice",
            "Mortgage Initial Escrow Account Disclosure Statement",
            "Mortgage Invoices",
            "Mortgage Land Lease or Land Trust",
            "Mortgage Land Title Adjustment",
            "Mortgage Last Will and Testament",
            "Mortgage Legal Description",
            "Mortgage Letters Of Administration",
            "Mortgage Letters of Testamentary",
            "Mortgage Listing Agreement",
            "Mortgage Litigation Guarantee",
            "Mortgage DIL Closing",
            "Mortgage Hardship Letter",
            "Mortgage Hardship Affidavit",
            "Mortgage Home Affordable Modification Agreement",
            "Mortgage Profit And Loss",
            "Mortgage Earnest Money Promissory Note",
            "Mortgage Rental Agreement",
            "Mortgage Repayment Plan",
            "Mortgage Short Sale Miscellaneous",
            "Mortgage LM - Trial Offer Letter or Plan",
            "Mortgage Errors and Omissions Agreement",
            "Mortgage Custom Type 2",
            "Mortgage Custom Type 1",
            "Mortgage Loan Agreement",
            "Mortgage Loan Closing Information Summary",
            "Mortgage Loan Modification",
            "Mortgage Loan Summary Report",
            "Mortgage Lock Confirmation",
            "Mortgage Loss Drafts",
            "Mortgage Loss Mitigation",
            "Mortgage Lost Assignment Affidavit",
            "Mortgage Mech Lien",
            "Mortgage Mediation",
            "Mortgage MI Claim Explanation of Benefits",
            "Mortgage MI Policy Cancellation Document",
            "Mortgage MI Repurchase Document",
            "Mortgage Miscellaneous Lien Release",
            "Mortgage Mobile Home Documentation",
            "Mortgage Monthly Activity Report",
            "Mortgage Deed of Trust-Recorded",
            "Mortgage PMI Disclosure",
            "Mortgage Payments",
            "Mortgage Deed of Trust-Unrecorded",
            "Mortgage Motion For Relief",
            "Mortgage Note",
            "Mortgage Note Affidavit",
            "Mortgage Note Endorsements",
            "Mortgage Notice Of Appearance",
            "Mortgage Notice of Default Filedrecorded",
            "Mortgage Notice of Final Cure",
            "Mortgage Notice of Levy",
            "Mortgage Notice of Payment Change",
            "Mortgage Notice of Right to Cancel",
            "Mortgage Notice of Sale",
            "Mortgage Notice of Second Lien",
            "Mortgage Notice of Servicing Transfer-Transferee",
            "Mortgage Notice of Servicing Transfer-Transferor",
            "Mortgage Notice of Termination",
            "Mortgage Notice to Quit",
            "Mortgage Objection to Claim",
            "Mortgage Processing and Underwriting Doc Set",
            "Mortgage Objection to Motion for Relief",
            "Mortgage Affidavit of Occupancy",
            "Mortgage Occupancy Agreement",
            "Mortgage Occupancy Termination Agreement",
            "Mortgage Ombudsman Documents",
            "Mortgage Owner Affidavit",
            "Mortgage Ownership and Encumbrances Report",
            "Mortgage Pay History External",
            "Mortgage Paystub",
            "Mortgage Payoff Demand Statement",
            "Mortgage PMI Certificate",
            "Mortgage Post Petition Fee Notices",
            "Mortgage Post Sale Documents",
            "Mortgage Power of Attorney-Recorded",
            "Mortgage Power of Attorney-Unrecorded",
            "Mortgage Closing Instructions",
            "Mortgage Preliminary Modification",
            "Mortgage Merged-Privacy Policy Notice-Title Policy - Privacy Policy-1098 Privacy Policy",
            "Mortgage Probate Court Order",
            "Mortgage Proof of Claim",
            "Mortgage Property Legal and Vesting Report",
            "Mortgage Property Management Agreement",
            "Mortgage Property Notices",
            "Mortgage Public Assistance",
            "Mortgage Record Owner and Lien Certificate",
            "Mortgage Recorded Satisfaction",
            "Mortgage Regfore Affidavit Executed",
            "Mortgage Release of Lis Pendens",
            "Mortgage REO Bids",
            "Mortgage REO Other",
            "Mortgage Form 26-1820 Report and Certificate of Loan Disbursement",
            "Mortgage Request for Verification of Rent or Mortgage",
            "Mortgage Request for Waiver of R.E. Tax Escrow Requirements",
            "Mortgage 1003",
            "Mortgage RMA Package",
            "Mortgage Sale Postponement",
            "Mortgage Sale or Milestone Rescission",
            "Mortgage Satisfaction of Judgement Tax Mortgage Liens",
            "Mortgage Security Agreement",
            "Mortgage Separation Agreement",
            "Mortgage Servicing Acquisition",
            "Mortgage Servicing Disclosure Statement",
            "Mortgage Short Payoffs",
            "Mortgage Signature-Name Affidavit",
            "Mortgage Assumption of Mortgage",
            "Mortgage SCRA Related Documents",
            "Mortgage Social Security Card or Customer ID",
            "Mortgage Soft Delete",
            "Mortgage Flood Hazard Determination Form",
            "Mortgage Stipulated Agreement",
            "Mortgage Subordination Agreement",
            "Mortgage Subordination Request Form",
            "Mortgage Appointment of Substitute Trustee",
            "Mortgage Merged-Real Estate Taxes-Tax Bill-Tax Certificate",
            "Mortgage Tax Certificate",
            "Mortgage Tax Record Information Sheet",
            "Mortgage Tax Liens",
            "Mortgage Tax Search",
            "Mortgage Third Party Authorization",
            "Mortgage Title Commitment-Equity or Property Report",
            "Mortgage Title Policy",
            "Mortgage Title Policy Endorsement",
            "Mortgage Title Search",
            "Mortgage Title Insurance Other",
            "Mortgage Transfer of Claim",
            "Mortgage Uniform Underwriting and Transmittal Summary",
            "Mortgage Trustee Sale Guarantee",
            "Mortgage UCC-1 Financing Statement",
            "Mortgage Others",
            "Mortgage Unknown",
            "Mortgage Utility Bill",
            "Mortgage Valuation Orders",
            "Mortgage Verification Document Set",
            "Mortgage Verification of Service for Military Home Buyers",
            "Mortgage W2",
            "Mortgage W9",
            "Mortgage Wire Transfer Instructions",
            "Mortgage Workmens Compensation",
            "Mortgage Writ of Possession",
            "Mortgage Cover Page",
            "Mortgage Barcode Page",
            "Mortgage Wisconsin Tax Escrow Option Notice",
            "Mortgage Hazard Insurance Declaration",
            "Mortgage Flood Insurance Declaration",
            "Mortgage Quitclaim Deed",
            "Mortgage Tax Deed",
            "Mortgage Warranty Deed",
            "Mortgage ALTA Settlement Statement",
            "Mortgage Home Inspection Waiver",
            "Mortgage Insurance Disclosure"]
            document_type_for_questionset = gr.Dropdown(choices=document_types, label="Select the Document Type")
            tag_for_questionset = gr.Textbox(label="Please provide a name for the question set. Ex: rwikd-dot-basic-questionset-20230707.")
            csv_file = gr.File(label="Load a csv - 2 columns with the headers as field, question",file_types=['.csv'],type='file')
            
        
            with gr.Row():
                status_for_loading_csv = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_csv = gr.Button("Upload data into the database").style(full_width=False)
        
    
    load_pdf.click(load_pdf_and_generate_embeddings, inputs=[pdf_doc, open_ai_key, relevant_pages], outputs=status)
    summarize_pdf.click(summarize_contents,outputs=summary)
    load_csv.click(load_csv_and_store_questionset_into_sqlite, inputs=[csv_file, document_type_for_questionset, tag_for_questionset], outputs=status_for_loading_csv)

    load_questionsets.click(retrieve_document_type_and_questionsettag_from_sqlite,outputs=questionsets)
    load_fields_and_questions.click(retrieve_fields_and_questions,questionsets,fields_and_questions)
    answers_for_predefined_question_set.click(answer_predefined_questions, questionsets, answers)

    convert_into_ocr.click(ocr_converter,image_pdf, ocr_pdf)
    submit_query.click(answer_query,input,output)


#Use this flavor of demo.launch if you need the app to have an admin page. The credentials to login in this case
#would be admin/lm0R!Rm0#97r
#demo.launch(auth=("admin", "lm0R!Rm0#97r"))
demo.launch(debug=True)




