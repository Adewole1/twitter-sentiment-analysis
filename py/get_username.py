import requests
import os
from os.path import join, dirname
from dotenv import load_dotenv

import get_user_id
from get_user_id import UserId

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

class UserNameText():
    
    def __init__(self, *args):
        super(UserNameText, self).__init__(*args)
        # load twitter keys and tokens from .env for security reasons

        self.bearer_token = os.environ.get('BEARER_TOKEN')
        # self.main()
        
    
    def get_params(self):
        # Tweet fields are adjustable.
        # Options include:
        # attachments, author_id, context_annotations,
        # conversation_id, created_at, entities, geo, id,
        # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
        # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
        # source, text, and withheld
        return {"max_results": 20, "exclude":["retweets", "replies"]}


    def bearer_oauth(self, r):
        """
        Method required by bearer token authentication.
        """

        r.headers["Authorization"] = f"Bearer {self.bearer_token}"
        r.headers["User-Agent"] = "v2UserTweetsPython"
        return r


    def connect_to_endpoint(self, url, params):
        response = requests.request("GET", url, auth=self.bearer_oauth)
        # print(response.status_code)
        if response.status_code != 200:
            raise Exception(
                "Request returned an error: {} {}".format(
                    response.status_code, response.text
                )
            )
        return response.json()


    def main(self, user):
        # url = self.create_url()
        users = user
        userClass = UserId()
        user_ids, user_names = userClass.connect(users)
        data = dict()
        for i, j in zip(user_ids, user_names):
            # print('\nFor {}:'.format(j))
            url = "https://api.twitter.com/2/users/{}/tweets?max_results=10&exclude=retweets,replies".format(i)
            params = self.get_params()
            json_response = self.connect_to_endpoint(url, params)
            # data = json.dumps(json_response, indent=4, sort_keys=True)
            text_data = list()
            
            # print(json_response['data'])
            
            # '''
            for i in range(0, len(json_response['data'])):
                text = json_response['data'][i]['text']
                text = text.replace("\n", "")
                # print("\n{}".format(text))
                text_data.append(text)
                
            data[j] = text_data
            # print(data)
        return data
            


if __name__ == "__main__":
    UserNameText()
    

# users
# stinewolverines
# Awesemo_Com
# AwesemoNBA