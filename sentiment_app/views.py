# sentiment_app.views

from django.http import HttpResponse, JsonResponse
from django.views.generic import TemplateView
from django.template.loader import render_to_string
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from get_username import UserNameText
import classifier
from classifier import sentiment
import sys
sys.modules['classifier'] = classifier

class IndexView(TemplateView):
    template_name = "index.html"
    
    
class AboutView(TemplateView):
    template_name = "about.html"
    
@csrf_exempt  
def search_user(request):
    state = request.GET.get('state')
    if state:
        if int(state)==1:
            usernames = request.GET.get('usernames')
            if usernames:
                userClass = UserNameText()
                data = userClass.main(usernames)
                
                for name in data.keys():
                    analysis = []
                    for text in data[name]:
                        analysis.append((text, sentiment(text)[0]))
                    data[name] = analysis
                    
                html_response = render_to_string('sentiment_app/user_result.html', context = {'result': data})
            
                return HttpResponse(html_response, status=200)
                    
        elif int(state) == 2:
            text = request.GET.get('text')
            if text:
                data = sentiment(text)
                
                html_response = render_to_string('sentiment_app/text_result.html', context = {'result': data})
            
                return HttpResponse(html_response, status=200)
    else:
        return render(request, 'sentiment_app/index.html',
                    {})
            