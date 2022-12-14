from rest_framework import generics
from .convert_to_wav import convert_to_wav
from rest_framework.response import Response
from rest_framework import status
from .models import AudioModel
from .serializers import AudioFileSerializer,TextSerializer
from rest_framework.parsers import MultiPartParser
import json

class AudioFileView(generics.GenericAPIView):
    serializer_class = AudioFileSerializer
    parser_classes = [MultiPartParser, ]
    queryset = AudioModel.objects.all()

    def post(self, request, *args, **kwargs):
        #ファイルを受け取る
        file_obj = request.data
        temp_audio_file = request.FILES['file']
        #ファイルをwavファイルに変換
        converted_temp_audio_file = convert_to_wav(temp_audio_file)
        file_obj['file'] = converted_temp_audio_file
        
        #ファイルを保存する
        serializer = AudioFileSerializer(data=file_obj)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        #ファイルを保存する
        model = serializer.save(commit=False)
        result = model.predicted_text()
        return Response(json.dumps({"result":result}), status=status.HTTP_201_CREATED)

class TextView(generics.GenericAPIView):
    serializer_class = TextSerializer

    def post(self, request, *args, **kwargs):
        #ファイルを受け取る
        serializer = TextSerializer(data={'text':request.data['data']})
        #ファイルが正しいか確認する
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        model = serializer.save(commit=False)
        #文章要約を行う
        _, summary = model.predict()
        return Response(json.dumps({"result":summary}), status=status.HTTP_201_CREATED)
