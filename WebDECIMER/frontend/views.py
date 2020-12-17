from django.shortcuts import render, HttpResponse


def index(request):
    return render(request, 'frontend/index.html')
    #return HttpResponse("My index should be here")
