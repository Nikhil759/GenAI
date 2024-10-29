import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock
import boto3
# import chardet
# from io import BytesIO
import tempfile


def hr_index():
    """
    This function does the below operations :
    1) Loads the PDF file from a given URL
    2) Splits it into chunks
    3) Creates a vector embedding using bedrock amazon.titan-embed-text-v1
    4) Stores the vector embedding in FAISS vector store
    5) Creates a vector index with the embedded data in vector store
    """


    data_load=PyPDFLoader("https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf")
    
    
    data_test=data_load.load_and_split()
    # print(len(data_test))
    # print(data_test[4])

    s3 = boto3.client('s3')
    bucket_name='rag-app1'
    file_key='Leave-Policy-India.pdf'
    response=s3.get_object(Bucket=bucket_name,Key=file_key)
    raw_data=response['Body'].read()
    # print(file_content)

    # detected_encoding=chardet.detect(raw_data)['encoding']

    # print(f"Detected Encoding : {detected_encoding}")

    # file_content=raw_data.decode(detected_encoding)
    # print(file_content)
    # Create a temporary file to save the PDF data
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(raw_data)
        temp_file_path = temp_file.name


    pdf_loader=PyPDFLoader(temp_file_path)
    documents=pdf_loader.load()

    data_split=RecursiveCharacterTextSplitter(separators=["\n\n","\n"," ", ""], chunk_size=100, chunk_overlap=10)
    # data_sample='''Guidelines:  
    # 1. An employee will be entitled to take PL after the line manager approval. 
    # 2. An employee can avail PL  not more than thrice  in a year, any deviation to this can be 
    # approved by the HRBP Head subject to manager approval. 
    # 3. For long leave, which is 7 days and more, the employee must inform at least one month in 
    # advance. Where an employee plans PL up to 7 days, he/she will be required to take approval 
    # at least 15 days in advance.  
    # 4. Extensions will not be granted, except in cases of emergencies. If an employee after 
    # proceeding on leave desires an extension thereof, he/she shall make an application in writing 
    # to the company for this purpose. 
    # 5. The Management reserves the right to reject or extend the leave at its discretion and without 
    # assigning any reason whatsoever. 
    # 6. When an employee takes a PL, weekend and other holidays will not be included while 
    # calculating leave. 
    # 7. In the event of the an employee resigning, retrenchment or termination of services by the 
    # company, unutilized privileged leave not exceeding 60 / 120 / 240 days as per his / her 
    # eligibility standing to his/her credit at the time of retirement, resignation, retrenchment or 
    # termination by the company, shall be allowed to be encashed, wherein encashment will be 
    # on monthly Gross Salary for all employees.'''
    # data_split_test=data_split.split_text(data_sample)
    # print(data_split_test)


    data_embeddings=BedrockEmbeddings(
        credentials_profile_name="bedrock_user",
        model_id="amazon.titan-embed-text-v1"
    )

    data_index=VectorstoreIndexCreator(text_splitter=data_split, embedding=data_embeddings, vectorstore_cls=FAISS)

    db_index=data_index.from_loaders([pdf_loader])

    # {
    #  "modelId": "amazon.titan-embed-text-v1",
    #  "contentType": "application/json",
    #  "accept": "*/*",
    #  "body": "{\"inputText\":\"this is where you place your input text\"}"
    # }

    return db_index

def hr_llm():
    """
    This function creates the Bedrock model
    """
    llm=Bedrock(
        credentials_profile_name="bedrock_user",
        model_id="meta.llama3-8b-instruct-v1:0",
    )

    return llm

    # {
    #  "modelId": "meta.llama3-8b-instruct-v1:0",
    #  "contentType": "application/json",
    #  "accept": "application/json",
    #  "body": "{\"prompt\":\"this is where you place your input text\",\"max_gen_len\":512,\"temperature\":0.5,\"top_p\":0.9}"
    # }

def hr_rag_response(index,question):
    """
    1) This function initializes the bedrock model
    2) It passes the input question to the vector index created in previous step, along with the initialized rag

    In this way the question is searched in the vector index and the rag_llm generates an answer using this vector index by searching the required info from the vector embedding
    """
    rag_llm=hr_llm()
    hr_rag_query=index.query(question=question,llm=rag_llm)
    return hr_rag_query