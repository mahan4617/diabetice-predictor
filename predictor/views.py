import numpy as np
from django.shortcuts import render
from .forms import DiabetesForm
from .ml_model import model, scaler


def landing_view(request):
    return render(request, 'predictor/landing.html')


def predict_view(request):
    prediction = None
    probability = None
    if request.method == 'POST':
        form = DiabetesForm(request.POST)
        if form.is_valid():
            cleaned = form.cleaned_data
            values = [
                cleaned['pregnancies'],
                cleaned['glucose'],
                cleaned['blood_pressure'],
                cleaned['skin_thickness'],
                cleaned['bmi'],
                cleaned['diabetes_pedigree_function'],
                cleaned['age'],
            ]
            user_features = np.array(values).reshape(1, -1)
            user_scaled = scaler.transform(user_features)
            pred = model.predict(user_scaled)[0]
            proba = model.predict_proba(user_scaled)[0][1]
            prediction = int(pred)
            probability = float(proba * 100.0)
    else:
        form = DiabetesForm()
    context = {
        'form': form,
        'prediction': prediction,
        'probability': probability,
    }
    return render(request, 'predictor/home.html', context)
