from django.shortcuts import render
from .models import RadioStation
from .utils.stream_manager import StreamManager

stream_manager = StreamManager()

def index(request):
    if request.method == "POST":
        station_id = request.POST.get("station_id")
        station = RadioStation.objects.get(id=station_id)
        stream_manager.add_stream(station.name, station.url)
    stations = RadioStation.objects.all()
    return render(request, 'streams/index.html', {'stations': stations})

def dashboard(request):
    return render(request, 'streams/dashboard.html')
