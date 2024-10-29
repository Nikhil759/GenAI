import os
from langchain.document_loaders import PyPDFLoader
import boto3
# import chardet
# from io import BytesIO
import tempfile


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

# print(documents)

# for i, doc in enumerate(documents):
#     print(f"Page {i + 1}:")
#     print(doc.page_content)

os.remove(temp_file_path)


data_load=PyPDFLoader("https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf")
print(data_load)
exit()
data_test=data_load.load_and_split()
print(len(data_test))
print(data_test[4])