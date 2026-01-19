from django import forms


class DiabetesForm(forms.Form):
    pregnancies = forms.IntegerField(min_value=0, label='Pregnancies')
    glucose = forms.FloatField(min_value=0, label='Glucose')
    blood_pressure = forms.FloatField(min_value=0, label='Blood Pressure')
    skin_thickness = forms.FloatField(min_value=0, label='Skin Thickness')
    bmi = forms.FloatField(min_value=0, label='BMI')
    diabetes_pedigree_function = forms.FloatField(min_value=0, label='Diabetes Pedigree Function')
    age = forms.IntegerField(min_value=1, label='Age')
