import sys
from django.shortcuts import render

# Add the path to arima_model.py before importing it
arima_model_path = 'C:\\Users\\DELL\\OneDrive - University of Arkansas\\Desktop\\ricepriceapp'
if arima_model_path not in sys.path:
    sys.path.append(arima_model_path)

import arima_model

# Create your views here.
def index(request):
    context = {}  # Initialize an empty context dictionary

    if request.method == 'POST':
        # Extract the date from the form submission
        date_input = request.POST.get('date_input')

        # Perform prediction using the arima_model
        predicted_price = arima_model.predict_rice_price(date_input)

        # Add the predicted price to the context
        context['predicted_price'] = predicted_price

    # Render the page with the context
    return render(request, 'index.html', context)