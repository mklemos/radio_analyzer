from django.shortcuts import render
from django.http import JsonResponse
from .models import RadioStation, Transcription
from .utils.stream_manager import StreamManager

stream_manager = StreamManager()

def index(request):
    if request.method == "POST":
        station_id = request.POST.get("station_id")
        station = RadioStation.objects.get(id=station_id)
        stream_manager.add_stream(station.name, station.url)  # Ensure this matches the model field name
    stations = RadioStation.objects.all()
    return render(request, 'streams/index.html', {'stations': stations})

def dashboard(request):
    return render(request, 'streams/dashboard.html')

def get_transcriptions(request):
    transcriptions = Transcription.objects.all().order_by('-timestamp')
    data = [{'station': t.station.name, 'timestamp': t.timestamp, 'text': t.text} for t in transcriptions]
    return JsonResponse(data, safe=False)
