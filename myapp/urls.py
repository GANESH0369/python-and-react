from .views import myapi, AudioToTextView
from django.urls import path
from .views import ConvertPDFToDOCX, ConvertDOCXToPDF, ConvertJPGToPDF,ConvertPDFToJPG, CompressPDF, png_to_text
from myapp import views
from .views import DownloadFileAPIView, PredictAPIView , PredictionView, UploadFileToS3


urlpatterns = [
    # path('convert/text', PNGToText.as_view(), name='convert_png_to_text'),
    path('',myapi.as_view(), name='myapi'),
    path('audio-to-text/', AudioToTextView.as_view(), name='audio-to-text'),
    path('convert/pdf-to-jpg', ConvertPDFToJPG.as_view(), name='convert_pdf_to_docx'),
    path('convert_jpg_to_pdf/', ConvertJPGToPDF.as_view(), name='convert_jpg_to_pdf'),
    path('convert/pdf-to-docx/', ConvertPDFToDOCX.as_view(), name='pdf_to_docx'),
    path('doc/', ConvertDOCXToPDF.as_view(), name='doc_to_pdf'),
    path('convert/compress',CompressPDF.as_view(),name='CompressPDF'),
    path('api/png-to-text/', png_to_text, name='png_to_text'),
    path('connect-db/', views.db_connect),
    # path('convert-text/', ConvertTextView.as_view(), name='convert_text'),
    # path('download/', S3FileDownloadView.as_view(), name='file-download'),
    path('download/', DownloadFileAPIView.as_view(), name='download_file'),
    path('predict/', PredictAPIView.as_view(), name='predict'),
    path('predictts/', PredictionView.as_view(), name='predict'),
    path('upload-file-into-s3/',UploadFileToS3.as_view(),name='upload-file-into-s3')
]



