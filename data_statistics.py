'''
Raw data statistics for the dataset:

Possible things to include
- Number of unique words (vocab size)
- Number of dialogues, scenes, episode
- Average number of words in each line
- Number of speakers and dialogues per speaker
- Most popular words
'''

import pickle
import helper



if __name__ = "__main__":

    data_dict = helper.load_data("data.pkl")
    token_dict = token_lookup()


'''
Example of how to parse:

episode = data_dict["series-7-episode-15"]
for line in episode:
    parts = line.split(":")
    speaker = parts[0].strip()
    dialogue = parts[1].strip()

    if speaker == "Scene":
        pass
'''
