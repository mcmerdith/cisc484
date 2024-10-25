import re
import time
from datetime import datetime, timedelta

url_pattern = re.compile(r"http\S+")
hashtag_pattern = re.compile(r"#\S+")
mention_pattern = re.compile(r"@\S+")
rule = re.compile(r"[^a-zA-Z0-9\s]")
# rule = re.compile(r'[^a-zA-Z\s]')
rule_all = re.compile(r"[^a-zA-Z0-9\s]")


class User:
    """
    TwitterUser class can used to save a user
    Including userbased features
    """

    # user_feature = []
    def __init__(self, tweet_json):
        self.user_json = tweet_json["user"]
        self.id_str = self.user_json["id_str"]
        self.screen_name = self.user_json["screen_name"]
        self.name = self.user_json["name"]
        self.description = self.user_json["description"] or ""
        self.verified = self.user_json["verified"]
        self.created_at = self.json_date_to_stamp(self.user_json["created_at"])
        self.followers_count = self.get_followers_count()
        self.friends_count = self.get_friends_count()
        self.statuses_count = self.get_statuses_count()

    def json_date_to_stamp(self, json_date):
        """
        exchange date from json format to timestamp(int)
        input:
            date from json
        output:
            int
        """
        time_strpt = time.strptime(json_date, "%a %b %d %H:%M:%S +0000 %Y")
        stamp = int(time.mktime(time_strpt))
        return stamp

    def json_date_to_os(self, json_date):
        """
        exchange date from json format to linux OS format
        input:
            date from json
        output:
            datetime
        """
        time_strpt = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.strptime(json_date, "%a %b %d %H:%M:%S +0000 %Y"),
        )
        os_time = datetime.strptime(str(time_strpt), "%Y-%m-%d %H:%M:%S")
        return os_time

    def get_name_length(self):
        return len(self.name)

    def get_screen_name_length(self):
        return len(self.screen_name)

    def get_description_length(self):
        return len(self.description)

    def get_verified(self):
        return "1" if self.verified else "0"

    def get_followers_count(self):
        """
        return follower count
        """
        followers_count = self.user_json["followers_count"]
        if followers_count == 0:
            followers_count = 1
        return followers_count

    def get_friends_count(self):
        """
        return friends count
        """
        friends_count = self.user_json["friends_count"]
        if friends_count == 0:
            friends_count = 1
        return friends_count

    def get_statuses_count(self):
        """
        return statuses count
        """
        statuses_count = self.user_json["statuses_count"]
        if statuses_count == 0:
            statuses_count = 1
        return statuses_count

    def get_user_age(self):
        """
        Age of an account
        get age feature of an account, remember call this function all the time. Time exchange
        """
        account_start_time = self.json_date_to_os(self.user_json["created_at"])
        now_time = datetime.now()
        account_age = (now_time - account_start_time).days
        if account_age == 0:
            account_age = 1
        return account_age

    def get_user_favourites(self):
        """
        get user favourites count
        """
        favourites_count = self.user_json["favourites_count"]
        return favourites_count


class Tweet:
    """
    TwitterTweet class can used to save a tweet
    """

    def __init__(self, tweet_json):
        self.tweet_json = tweet_json
        self.user = User(tweet_json)
        if "retweeted_status" in tweet_json.keys():
            self.retweet = Tweet(tweet_json["retweeted_status"])
        else:
            self.retweet = None

        self.text = tweet_json["text"]
        self.timestr = self.json_date_to_stamp(tweet_json["created_at"])
        self.tid_str = self.tweet_json["id_str"]
        self.entities = self.tweet_json["entities"]

    def json_date_to_stamp(self, json_date):
        """
        exchange date from json format to timestamp(int)
        input:
            date from json
        output:
            int
        """
        time_strpt = time.strptime(json_date, "%a %b %d %H:%M:%S +0000 %Y")
        stamp = int(time.mktime(time_strpt))
        return stamp

    def is_en(self):
        return self.tweet_json["lang"] == "en"

    def get_id(self):
        return self.tid_str

    def get_hashtag_count(self):
        """
        get number of hashtags
        """
        hashtags = self.tweet_json["entities"]["hashtags"]
        return len(hashtags)

    def get_text_len(self):
        """
        return chars of text
        """
        return len(self.text)

    def get_retweet_user_id(self):
        if self.retweet is None:
            return "0"
        else:
            return self.retweet.user.id_str
    
    def get_retweeted(self):
        return "1" if self.retweeted_status else "0"
    
    def get_url_count(self):
        return len(self.entities["urls"])
    
    def get_media_count(self):
        if "media" in self.entities:
          return len(self.entities["media"])
        else:
          return "0"

    def get_tweet_features(self):
        feature_list = []
        feature_list.append(self.user.get_user_age())  # 1
        feature_list.append(self.user.get_user_favourites())  # 3
        feature_list.append(self.get_hashtag_count())  # 8
        feature_list.append(self.get_text_len())  # 11
        # feature_list.append(self.user.get_name_length())
        # feature_list.append(self.user.get_screen_name_length())
        # feature_list.append(self.user.get_description_length())
        feature_list.append(self.get_url_count())
        feature_list.append(self.get_media_count())
        feature_list.append(self.user.get_verified())
        feature_list.append(self.user.get_followers_count())
        feature_list.append(self.user.get_friends_count())
        feature_list.append(self.user.get_statuses_count())
        # feature_list.append(self.get_retweet_user_id())

        return feature_list
