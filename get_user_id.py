import requests
import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


class UserId():
    
    def __init__(self, *args):
        super(UserId, self).__init__(*args)
        
        # load twitter keys and tokens from .env for security reasons
        self.bearer_token = os.environ.get('BEARER_TOKEN')
        # self.connect()

    def create_url(self, user):
        # Specify the usernames that you want to lookup below
        # You can enter up to 100 comma-separated values.
        users = user.split(',')
        separator = ','
        usernames = "usernames={}".format(separator.join(users))
        # print(usernames)
        # usernames = "usernames=TwitterDev,TwitterAPI"
        user_fields = "user.fields=description,created_at"
        # User fields are adjustable, options include:
        # created_at, description, entities, id, location, name,
        # pinned_tweet_id, profile_image_url, protected,
        # public_metrics, url, username, verified, and withheld
        url = "https://api.twitter.com/2/users/by?{}&{}".format(usernames, user_fields)
        return url


    def bearer_oauth(self, r):
        """
        Method required by bearer token authentication.
        """

        r.headers["Authorization"] = f"Bearer {self.bearer_token}"
        r.headers["User-Agent"] = "v2UserLookupPython"
        return r


    def connect_to_endpoint(self, url):
        response = requests.request("GET", url, auth=self.bearer_oauth,)
        # print(response.status_code)
        if response.status_code != 200:
            raise Exception(
                "Request returned an error: {} {}".format(
                    response.status_code, response.text
                )
            )
        return response.json()


    def connect(self, user):
        url = self.create_url(user)
        json_response = self.connect_to_endpoint(url)
        response = json_response['data']
        ids = []
        names = []
        for i in range(0, len(response)):
            ids.append(response[i]['id'])
            names.append(response[i]['name'])
            # print('\n',response[i]['id'])
        # for i, j in zip(ids,names):
        #     print("Id is {} with username:{}".format(i, j))
        return ids, names

if __name__ == "__main__":
    UserId()