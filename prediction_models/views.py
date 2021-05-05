from django.shortcuts import render
from rest_framework.decorators import api_view
from prediction_models.controllers.predict_model import control_request

# Create your views here.
@api_view(['GET'])
def predict_models(request):
    return control_request(request.data)