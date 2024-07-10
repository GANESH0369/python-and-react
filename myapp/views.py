from rest_framework.views import APIView
from .models import emp_data
from .serilizers import seril
from rest_framework.response import Response
from rest_framework import status
from .utils import convert_audio_to_text
from pdf2docx import Converter
from docx2pdf import convert
import os
from tempfile import NamedTemporaryFile
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet




class myapi(APIView):
    def get(self,request):
        mydata=emp_data.objects.all()
        seriu=seril(mydata,many=True)
        
        return Response(seriu.data)
    def post(self,request):
        seriu=seril(data=request.data)
        if seriu.is_valid():
            seriu.save()
            return Response(seriu.data, status=status.HTTP_201_CREATED)
        return Response(seriu.errors, status=status.HTTP_400_BAD_REQUEST)


class AudioToTextView(APIView):
    def post(self, request):
        if request.FILES.get('audio_file'):
            audio_file = request.FILES['audio_file']
            result = convert_audio_to_text(audio_file)
            return Response({'Your text': result}, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Please provide an audio file.'}, status=status.HTTP_400_BAD_REQUEST)
        

# class ConvertPDFToDOCX(APIView):
#     def post(self, request, format=None):
#         if 'file' not in request.FILES:
#             return Response({'error': 'No file was uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        
#         pdf_file = request.FILES['file']
        
#         try:
#             with open(pdf_file.temporary_file_path(), 'rb') as pdf:
#                 cv = Converter(pdf)
#                 docx_path = r"C:\New folder\converted_file.docx" 
#                 cv.convert(docx_path)
#                 cv.close()
#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
#         return Response({'docx_path': docx_path}, status=status.HTTP_200_OK)



class ConvertPDFToDOCX(APIView):
    def post(self, request, format=None):
        if 'file' not in request.FILES:
            return Response({'error': 'No file was uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        
        pdf_file = request.FILES['file']
        
        with NamedTemporaryFile(delete=False) as temp_pdf:
            for chunk in pdf_file.chunks():
                temp_pdf.write(chunk)
            pdf_path = temp_pdf.name
        
        docx_path = os.path.join(r"C:\New folder", 'Ganesh.docx')
        
        try:
            with open(pdf_path, 'rb') as pdf:
                cv = Converter(pdf)
                cv.convert(docx_path)
                cv.close()
        except Exception as e:
            os.remove(pdf_path)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        os.remove(pdf_path)
        
        return Response({'docx_path': docx_path}, status=status.HTTP_200_OK)



class ConvertDOCXToPDF(APIView):
    def post(self, request, format=None):
        if 'file' not in request.FILES:
            return Response({'error': 'No file was uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        
        docx_file = request.FILES['file']
        
        with NamedTemporaryFile(delete=False) as temp_docx:
            for chunk in docx_file.chunks():
                temp_docx.write(chunk)
            docx_path = temp_docx.name
        
        pdf_path = os.path.join(r"C:\New folder", 'converted.pdf')
        
        try:
            # Read DOCX file
            doc = Document(docx_path)
            
            # Create PDF from DOCX content
            styles = getSampleStyleSheet()
            elements = []
            for paragraph in doc.paragraphs:
                text = paragraph.text
                if text:
                    p = Paragraph(text, styles["Normal"])
                    elements.append(p)
            
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            doc.build(elements)
            
        except Exception as e:
            os.remove(docx_path)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        os.remove(docx_path)
        
        return Response({'pdf_path': pdf_path}, status=status.HTTP_200_OK)


import os
from tempfile import NamedTemporaryFile
from pdf2image import convert_from_path

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

class ConvertPDFToJPG(APIView):
    def post(self, request, format=None):
        if 'file' not in request.FILES:
            return Response({'error': 'No file was uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        
        pdf_file = request.FILES['file']
        
        with NamedTemporaryFile(delete=False) as temp_pdf:
            for chunk in pdf_file.chunks():
                temp_pdf.write(chunk)
            pdf_path = temp_pdf.name
        
        try:
            # Convert PDF to JPG images
            images = convert_from_path(pdf_path)
        except Exception as e:
            os.remove(pdf_path)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        os.remove(pdf_path)
        
        # Save JPG images to temporary files
        jpg_paths = []
        for i, image in enumerate(images):
            jpg_path = os.path.join(r"C:\\New folder", f'page_{i+1}.jpg')
            image.save(jpg_path, 'JPEG')
            jpg_paths.append(jpg_path)
        
        return Response({'jpg_paths': jpg_paths}, status=status.HTTP_200_OK)
    
# ---------------------------------------------
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views import View
from django.core.files.uploadedfile import InMemoryUploadedFile
from tempfile import NamedTemporaryFile
from pdf2image import convert_from_path
import os

@method_decorator(csrf_exempt, name='dispatch')
class ConvertJPGToPDF(View):
    def post(self, request, *args, **kwargs):
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file was uploaded'}, status=400)
        
        jpg_file = request.FILES['file']
        
        # Check if the uploaded file is a JPG
        if not jpg_file.name.endswith('.jpg'):
            return JsonResponse({'error': 'The uploaded file is not a JPG'}, status=400)

        # Save the uploaded JPG temporarily
        with NamedTemporaryFile(delete=False, suffix='.jpg') as temp_jpg:
            for chunk in jpg_file.chunks():
                temp_jpg.write(chunk)
            jpg_path = temp_jpg.name
        
        # Path to save the PDF
        pdf_path = os.path.join(r"C:\\New folder", 'convertedh.pdf')
        
        try:
            # Convert JPG to PDF
            pages = convert_from_path(jpg_path)
            for page in pages:
                page.save(pdf_path, 'PDF', resolution=100.0, save_all=True)
        except Exception as e:
            os.remove(jpg_path)
            return JsonResponse({'error': str(e)}, status=500)
        
        os.remove(jpg_path)
        
        return JsonResponse({'pdf_path': pdf_path}, status=200)


# ----------------------


class ConvertPDFToJPG(APIView):
    def post(self, request, format=None):
        if 'file' not in request.FILES:
            return Response({'error': 'No file was uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        
        pdf_file = request.FILES['file']
        
        with NamedTemporaryFile(delete=False) as temp_pdf:
            for chunk in pdf_file.chunks():
                temp_pdf.write(chunk)
            pdf_path = temp_pdf.name
        
        try:
            # Convert PDF to JPG images
            images = convert_from_path(pdf_path)
        except Exception as e:
            os.remove(pdf_path)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        os.remove(pdf_path)
        
        # Save the JPG images and return their paths
        jpg_paths = []
        for i, image in enumerate(images):
            jpg_path = os.path.join(r"C:\New folder", f'converted_page_{i+1}.jpg')
            image.save(jpg_path)
            jpg_paths.append(jpg_path)
        
        return Response({'jpg_paths': jpg_paths}, status=status.HTTP_200_OK)


import os
from tempfile import NamedTemporaryFile
from PyPDF2 import PdfReader, PdfWriter

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

class CompressPDF(APIView):
    def post(self, request, format=None):
        if 'file' not in request.FILES:
            return Response({'error': 'No file was uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        
        pdf_file = request.FILES['file']
        
        with NamedTemporaryFile(delete=False) as temp_pdf:
            for chunk in pdf_file.chunks():
                temp_pdf.write(chunk)
            pdf_path = temp_pdf.name
        
        compressed_pdf_path = os.path.join(r"C:\New folder", 'compressed.pdf')
        
        try:
            # Compress PDF
            with open(pdf_path, 'rb') as original_pdf:
                reader = PdfReader(original_pdf)
                writer = PdfWriter()

                # Copy pages from the original PDF to the compressed PDF
                for page_num in range(len(reader.pages)):
                    writer.add_page(reader.pages[page_num])

                # Write the compressed PDF to disk
                with open(compressed_pdf_path, 'wb') as compressed_pdf:
                    writer.write(compressed_pdf)
        except Exception as e:
            os.remove(pdf_path)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        os.remove(pdf_path)
        
        return Response({'compressed_pdf_path': compressed_pdf_path}, status=status.HTTP_200_OK)


# views.py
# import pytesseract
# from PIL import Image
# from rest_framework.response import Response
# from rest_framework.views import APIView

# class PNGToText(APIView):
#     def post(self, request):
#         # Assuming the PNG image is sent as 'file' in the request
#         uploaded_file = request.FILES['file']
        
#         # Open the image using PIL (Python Imaging Library)
#         image = Image.open(uploaded_file)
        
#         # Use pytesseract to perform OCR (Optical Character Recognition)
#         text = pytesseract.image_to_string(image)
        
#         # Return the extracted text as a response
#         return Response({'text': text})


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import pytesseract
from io import BytesIO
# import sqlalchemy_hana


@csrf_exempt
def png_to_text(request):
    if request.method == 'POST':
        # Ensure request has the image data
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']

            # Open the image using PIL
            img = Image.open(uploaded_file)

            # Perform OCR using pytesseract
            text = pytesseract.image_to_string(img)

            # Return the text as response
            return JsonResponse({'text': text})
        else:
            return JsonResponse({'error': 'No file found in request'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)


# --------------------------------------------------------------------------------------

from django.http import JsonResponse
from hdbcli.dbapi import connect

def db_connect(request):
    try:
        # Replace these variables with your actual HANA database credentials
        hana_host = 'a67278e3-b6fe-4983-b9cb-2271f0fdc823.hna0.prod-us10.hanacloud.ondemand.com'
        hana_port =443
        hana_user = 'DBADMIN'
        hana_password = 'Mouri@2024'

        connection = connect(
            user=hana_user,
            password=hana_password,
            address=hana_host,
            port=int(hana_port)
        )

        # Close the connection
        connection.close()

        return JsonResponse({"message": "Database connection successful"})
    except Exception as e:
        return JsonResponse({
            "Error": "Failed to connect with SAP database",
            "Error message": str(e),
            "Error code": 400
        })


# # from gensim.models import KeyedVectors
# import gensim.downloader as api
# from gensim.models import Word2Vec


# word2vec_model = api.load("word2vec-google-news-300")

# def text_to_embedding(text):
#     words = text.split()
#     embedding = []
#     for word in words:
#         if word in word2vec_model:
#             embedding.append(word2vec_model[word].tolist())  # Convert to list for JSON serialization
#     return embedding

# class ConvertTextView(APIView):
#     def post(self, request, *args, **kwargs):
#         data = request.data
#         text = data.get('text')
#         if text:
#             embedding = text_to_embedding(text)
#             return Response({'embedding': embedding}, status=status.HTTP_200_OK)
#         else:
#             return Response({'error': 'Text not provided'}, status=status.HTTP_400_BAD_REQUEST)



# # from gensim.models import Word2Vec
# import gensim.downloader as api

# class ConvertTextView(APIView):
#     def post(self, request, *args, **kwargs):
#         data = request.data
#         text = data.get('text')
#         if text:
#             embedding = self.text_to_embedding(text)
#             return Response({'embedding': embedding}, status=status.HTTP_200_OK)
#         else:
#             return Response({'error': 'Text not provided'}, status=status.HTTP_400_BAD_REQUEST)

#     def text_to_embedding(self, text):
#         word2vec_model = api.load("word2vec-google-news-300")
#         words = text.split()
#         embedding = []
#         for word in words:
#             if word in word2vec_model:
#                 embedding.append(word2vec_model[word].tolist())  # Convert to list for JSON serialization
#         return embedding



# import boto3
# from django.conf import settings

# class S3FileDownloadView(APIView):
#     """
#     View to download a file from AWS S3.
#     """
#     def get(self, request, filename):
#         # Create an S3 client
#         s3 = boto3.client('s3', aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
#                           aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
#                           region_name='us-east-1')  # Adjust the region as necessary

#         bucket_name = 'al-ml-models-data'

#         try:
#             # Attempt to get the object from S3
#             response = s3.get_object(Bucket=bucket_name, Key=filename)
#             file_content = response['Body'].read()
#             # path=d//
#             # Set the content type to the file's content type stored in S3
#             content_type = response.get('ContentType', 'application/octet-stream')

#             return Response(file_content, content_type=content_type, status=status.HTTP_200_OK)

#         except s3.exceptions.NoSuchKey:
#             return Response({'message': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)
#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



# import boto3
# from django.conf import settings
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status

# class S3FileDownloadView(APIView):
#     """
#     View to download a file from AWS S3.
#     """
#     def post(self, request, filename):
#         # Create an S3 client
#         s3 = boto3.client('s3', aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
#                           aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
#                           region_name='us-east-1')  # Adjust the region as necessary

#         bucket_name = 'al-ml-models-data'

#         try:
#             # Attempt to get the object from S3
#             response = s3.get_object(Bucket=bucket_name, Key=filename)
#             file_content = response['Body'].read()
#             # Specify the local file path where you want to save the downloaded file
#             local_file_path = request.data.get("D:\\S3_file_access")
#             if local_file_path:
#                 with open(local_file_path, 'wb') as local_file:
#                     local_file.write(file_content)
#                 return Response({'message': 'File downloaded successfully.'}, status=status.HTTP_200_OK)
#             else:
#                 return Response({'message': 'Please provide local_file_path in the request body.'}, status=status.HTTP_400_BAD_REQUEST)

#         except s3.exceptions.NoSuchKey:
#             return Response({'message': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)
#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




# # views.py
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from boto3.session import Session
# import os

# class DownloadFileAPIView(APIView):
#     def get(self, request):
#         # AWS S3 configuration
#         aws_access_key_id = 'your-access-key-id'
#         aws_secret_access_key = 'your-secret-access-key'
#         bucket_name = 'your-bucket-name'
#         file_key = 'path/to/your/file'

#         # Temporary directory to store downloaded file
#         temp_dir = '/tmp'  # Change this to your preferred directory

#         # Initialize AWS session
#         session = Session(
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key
#         )
#         s3 = session.resource('s3')

#         try:
#             # Download file from S3 bucket to local machine
#             file_path = os.path.join(temp_dir, os.path.basename(file_key))
#             s3.Bucket(bucket_name).download_file(file_key, file_path)
#             return Response({'message': f'File downloaded to {file_path}'})
#         except Exception as e:
#             return Response({'error': str(e)}, status=400)


# # views.py

# # import os
# import boto3
# from django.conf import settings
# class DownloadFileAPIView(APIView):
#     def get(self, request):
#         s3 = boto3.client('s3', 
#                           aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
#                           aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
#                           region_name='us-east-1')
        
#         # AWS S3 bucket and file details
#         bucket_name = 'al-ml-models-data'
#         file_key = 'ganesh-report-07-May-2024.pdf'  # File key without 's3://' prefix
        
#         temp_dir = "D:\\S3_file_access"  
#         try:
#             # Download file from S3 bucket to local machine
#             file_path = os.path.join(temp_dir, file_key)
#             s3.download_file(bucket_name, file_key, file_path)
#             return Response({'message': f'File downloaded to {file_path}'})
#         except Exception as e:
#             return Response({'error': str(e)}, status=400)
# # 


import os
import boto3
from django.conf import settings


class DownloadFileAPIView(APIView):
    def get(self, request):
        # Retrieve file_key from request parameters
        file_key = request.query_params.get('file_key')
        if not file_key:
            return Response({'error': 'File key is required'}, status=400)

        # Initialize the S3 client
        s3 = boto3.client(
            's3', 
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name='us-east-1'
        )
        
        # AWS S3 bucket details
        bucket_name = 'al-ml-models-data'
        
        # Temporary directory on local machine for file download
        temp_dir = r"D:\\4-08-django"
        try:
            # Full path where the file will be saved locally
            file_path = os.path.join(temp_dir, file_key)
            # Download file from S3 bucket to the specified local path
            s3.download_file(bucket_name, file_key, file_path)
            return Response({'message': f'File downloaded to {file_path}'})
        except Exception as e:
            return Response({'error': str(e)}, status=400)
        
import joblib

class PredictAPIView(APIView):
    def post(self, request):
        try:
            # Deserialize the incoming data
            data = request.data

            # Check if 'input' key is in the data
            if 'input' not in data:
                return Response({'error': 'No input data provided'}, status=status.HTTP_400_BAD_REQUEST)

            # Load the model
            model = joblib.load('D:\\path_to_your_model\\your_model_file.pkl') 
            # Make predictions
            predictions = model.predict([data['input']])

            # Return predictions as JSON response
            return Response({'predictions': predictions.tolist()})
        
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        


# import os
# import boto3
# from django.conf import settings
# from rest_framework.response import Response
# from rest_framework.views import APIView
# import joblib

# class DownloadFileAPIView(APIView):
#     def get(self, request):
#         file_key = request.query_params.get('file_key')
#         if not file_key:
#             return Response({'error': 'File key is required'}, status=400)

#         s3 = boto3.client('s3')
#         bucket_name = 'al-ml-models-data'

#         try:
#             obj = s3.get_object(Bucket=bucket_name, Key=file_key)
#             response = Response(obj['Body'].read(), content_type=obj['ContentType'])
#             response['Content-Disposition'] = f'attachment; filename="{file_key}"'
#             return response
#         except boto3.exceptions.S3UploadFailedError as e:
#             return Response({'error': str(e)}, status=400)
#         except Exception as e:
#             return Response({'error': str(e)}, status=500)

# class PredictAPIView(APIView):
#     def post(self, request):
#         try:
#             data = request.data
#             if 'input' not in data:
#                 return Response({'error': 'No input data provided'}, status=400)

#             model_path = os.environ.get('MODEL_FILE_PATH')
#             if not model_path:
#                 raise ValueError('MODEL_FILE_PATH environment variable is not set')

#             model = joblib.load(model_path)
#             predictions = model.predict([data['input']])
#             return Response({'predictions': predictions.tolist()})
        
#         except ValueError as e:
#             return Response({'error': str(e)}, status=400)
#         except Exception as e:
#             return Response({'error': str(e)}, status=500)

# # views.py
# import joblib
# from django.views.generic.edit import FormView
# from django.http import JsonResponse
# from .models import DataPoint
# from .forms import PredictionForm

# # Load the pre-trained ML model
# ml_model = joblib.load('ml_model.pkl')

# class PredictionView(FormView):
#     form_class = PredictionForm

#     def form_valid(self, form):
#         feature1 = form.cleaned_data['feature1']
#         feature2 = form.cleaned_data['feature2']
        
#         # Make prediction
#         prediction = ml_model.predict([[feature1, feature2]])[0]
        
#         # Save to database
#         DataPoint.objects.create(feature1=feature1, feature2=feature2, prediction=prediction)
        
#         # Return prediction as JSON response
#         return JsonResponse({'prediction': prediction})
    
#     def form_invalid(self, form):
#         # Return validation errors as JSON response
#         return JsonResponse(form.errors, status=400)


import joblib
from django.views import View
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.forms.models import model_to_dict
import json
from .models import DataPoint
from .forms import PredictionForm

# Load the pre-trained ML model
ml_model = joblib.load('ml_model.pkl')

@method_decorator(csrf_exempt, name='dispatch')
class PredictionView(View):
    
    def post(self, request, *args, **kwargs):
        try:
            # breakpoint()
            data = json.loads(request.body)
            print(data)
        except json.JSONDecodeError as e:
            return JsonResponse({'error': 'Invalid JSON input', 'details': str(e)}, status=400)

        form = PredictionForm(data)

        if form.is_valid():
            feature1 = form.cleaned_data['feature1']
            feature2 = form.cleaned_data['feature2']
            
            # Make prediction
            prediction = ml_model.predict([[feature1, feature2]])[0]
            
            # Save to database
            data_point = DataPoint.objects.create(
                feature1=feature1,
                feature2=feature2,
                prediction=prediction
            )
            
            # Return prediction as JSON response
            response_data = model_to_dict(data_point)
            return JsonResponse(response_data)
        
        # Include form errors in the response for debugging
        return JsonResponse({'error': 'Invalid form data', 'details': form.errors}, status=400)


import logging
import boto3
from botocore.exceptions import ClientError
import os
from django.http import JsonResponse
from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser, FormParser

class UploadFileToS3(APIView):
    # parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        file_obj = request.FILES['file']
        bucket = request.data.get('bucket')
        object_name = request.data.get('object_name')

        if not bucket or not file_obj:
            return JsonResponse({'error': 'Bucket name and file are required.'}, status=400)

        if object_name is None:
            object_name = os.path.basename(file_obj.name)

        # Upload the file
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_fileobj(file_obj, bucket, object_name)
        except ClientError as e:
            logging.error(e)
            return JsonResponse({'error': str(e)}, status=500)
        return JsonResponse({'message': 'File uploaded successfully.'}, status=200)
