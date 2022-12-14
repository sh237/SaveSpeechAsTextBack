from rest_framework import generics
from .convert_to_wav import convert_to_wav
from rest_framework.generics import (UpdateAPIView)
from rest_framework.response import Response
from rest_framework import status
from .models import AudioModel
from rest_framework import permissions
from .serializers import AudioFileSerializer,TextSerializer
from rest_framework.parsers import MultiPartParser, FormParser,FileUploadParser
from django.db import models
import os
from google.cloud import speech
from google.cloud import storage
from pydub import AudioSegment
import subprocess
import requests
from google.protobuf.json_format import MessageToJson
import json

class AudioFileView(generics.GenericAPIView):
    serializer_class = AudioFileSerializer
    parser_classes = [MultiPartParser, ]
    queryset = AudioModel.objects.all()

    def post(self, request, *args, **kwargs):
        print("post")
        file_obj = request.data
        print(request .data)
        temp_audio_file = request.FILES['file']
        print("temp_audio_file",temp_audio_file)
        # Using our custom convert_to_mp3 function to obtain converted file
        converted_temp_audio_file = convert_to_wav(temp_audio_file)
        
        # Adding this file to the serializer
        file_obj['file'] = converted_temp_audio_file
        print(file_obj)
        print(type(file_obj['file']))
        print(file_obj['file'].name)
        # AudioModel.objects.create(file=file_obj['file'])
        
        serializer = AudioFileSerializer(data=file_obj)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Actual place where we save it to the MEDIA_ROOT (cloud or other)
        model = serializer.save()
        print(model.file)
        result = model.predicted_text()
        

        return Response(json.dumps({"result":result}), status=status.HTTP_201_CREATED)

class TextView(generics.GenericAPIView):
    serializer_class = TextSerializer

    def post(self, request, *args, **kwargs):
        print("post")
        print("request.data['data']")
        print(request.data['data'])
        print(type(request.data['data']))
        serializer = TextSerializer(data={'text':request.data['data']})
        print(serializer)
        print(serializer.is_valid())
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Actual place where we save it to the MEDIA_ROOT (cloud or other)
        model = serializer.save()
        _, summary = model.predict()
        return Response(json.dumps({"result":summary}), status=status.HTTP_201_CREATED)
