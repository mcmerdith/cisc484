import datetime
import json
import os

from tool.Twitter import Tweet

start_date = "20201117"
end_date = "20210521"
start_date_datetime = datetime.datetime.strptime(start_date, "%Y%m%d")
end_date_datetime = datetime.datetime.strptime(end_date, "%Y%m%d")
proc_date = start_date_datetime
duration = 300  # t

data_check_list = os.listdir("Data/")
data_check_dic = {i: 1 for i in data_check_list}

for _ in range(duration):
    # process the data in this date
    proc_date_str = proc_date.strftime("%Y-%m-%d")

    input_data_folder_path = "Data/" + proc_date_str + "/"
    output_data_folder_path = "Tmp/" + proc_date_str + "/"

    if not proc_date_str in data_check_dic.keys():
        proc_date = proc_date + datetime.timedelta(days=1)
        if proc_date == end_date_datetime:
            break
        continue

    if not os.path.exists(output_data_folder_path):
        os.makedirs(output_data_folder_path)

    output_data_file = output_data_folder_path + "tweet_feature"
    with open(output_data_file, "w", encoding="utf-8") as file_out:

        for filename in os.listdir(input_data_folder_path):
            input_data_path = input_data_folder_path + filename

            with open(
                input_data_path, "r", encoding="utf-8", errors="ignore"
            ) as file_in:

                for line in file_in:
                    try:
                        tweet = json.loads(line)
                        tweet_obj = Tweet(tweet)
                    except:
                        print(tweet)
                        continue

                    if not tweet_obj.is_en():
                        continue

                    tweet_id = tweet_obj.get_id()
                    user_id = tweet_obj.user.id_str
                    feature_list = tweet_obj.get_tweet_features()
                    file_out.write(tweet_id + "\t" + user_id + "\t")

                    for f in feature_list:
                        file_out.write(str(f))
                        file_out.write("\t")
                    file_out.write("\n")
                    file_out.flush()

    proc_date = proc_date + datetime.timedelta(days=1)
    if proc_date == end_date_datetime:
        break
