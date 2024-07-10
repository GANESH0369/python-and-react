from django import forms

class PredictionForm(forms.Form):
    feature1 = forms.FloatField(label='Feature 1')
    feature2 = forms.FloatField(label='Feature 2')
