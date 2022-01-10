# sentiment_app.views

from django.views.generic import TemplateView
from django.shortcuts import render
from get_username import UserNameText

class IndexView(TemplateView):
    template_name = "index.html"
    
class AboutView(TemplateView):
    template_name = "about.html"
    
def search_user(request):
    if request.method == "POST":
        searched = request.POST['searched']
        userClass = UserNameText()
        result = userClass.main(searched)
        
        return render(request, 'sentiment_app/index.html',
                      {'searched':searched, 'result':result})
        
    else:
        return render(request, 'sentiment_app/index.html',
                      {'searched':'', 'result':''})
