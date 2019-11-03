from django import forms

class RequestForm(forms.Form):
    metadata = forms.CharField(label='metadata',max_length=10000)
    image = forms.ImageField()
