'''
Raw data statistics for the dataset:

Possible things to include
- Most popular words
'''

import pickle
import helper
import heapq
from collections import Counter
import numpy as np
import networkx as nx

def raw_stats(data_dict, vocab_to_int, int_to_vocab, token_dict):
    print("The total number of seasons is: ", 10)
    print("The total number of episodes is: ", len(data_dict))

    print("The number of punctuation considered is:", len(token_dict))
    print("The total number of unique words is: ", len(vocab_to_int))

    dialogueCounts = {}
    sceneCounts = {}
    wordCounts = {}

    linesPerScene = []
    wordsPerLine = []

    totalLines = 0
    totalScenes = 0

    for episode in data_dict:
        lines_in_current_scene = None
        for line in data_dict[episode]:
            totalLines += 1

            line = line.lower()
            parts = line.split(":")
            speaker = parts[0].split("(")[0].strip()
            dialogue = parts[1].strip()

            if lines_in_current_scene is None:
                totalScenes += 1
                lines_in_current_scene = 0
            elif speaker in ["scene", "sceme", "caption"]:
                totalScenes += 1
                linesPerScene.append(lines_in_current_scene)
                lines_in_current_scene = 0
            else:
                lines_in_current_scene += 1
                prev_count = dialogueCounts.get(speaker, 0)
                dialogueCounts[speaker] = prev_count + 1

            # replace punctuations with special tokens
            for key, token in token_dict.items():
                line = line.replace(key, ' {} '.format(token))
            words = line.split()
            wordsPerLine.append(len(words))
            for word in words:
                if word not in token_dict.values():
                    wordCounts[word] = wordCounts.get(word, 0) + 1

    print("The total number of scenes is:", totalScenes)
    print("The total number of lines is:", totalLines)
    print("The average number of tokens per line is:", sum(wordsPerLine)/len(wordsPerLine))
    print("The average number of lines per scene is:", sum(linesPerScene)/len(linesPerScene))

    print("The number of speakers is:", len(dialogueCounts))
    print("The number of dialogues for the most popular speakers are:",  dict(Counter(dialogueCounts).most_common(10)))

    print("Top occuring words", dict(Counter(wordCounts).most_common(25)))



def degree_centrality(data_dict, vocab_to_int, int_to_vocab, token_dict):

    degreeDict = {}

    for episode in data_dict:
        for line in data_dict[episode]:
            line = line.lower()

            parts = line.split(":")
            speaker = parts[0].split("(")[0].strip()
            dialogue = parts[1].strip()

            if speaker in ["sheldon", "leonard", "penny", "howard", "raj", "amy", "bernadette"]:

                # replace punctuations with special tokens
                for key, token in token_dict.items():
                    dialogue = dialogue.replace(key, ' {} '.format(token))

                words = dialogue.split()
                for word in words:
                    if word in ["sheldon", "leonard", "penny", "howard", "raj", "amy", "bernadette"]:
                        degreeDict[speaker] = degreeDict.get(speaker, {})
                        degreeDict[speaker][word] = degreeDict[speaker].get(word, 0) + 1


    # convert dict into numpy array
    degreeList = []
    for speaker in ["sheldon", "leonard", "penny", "howard", "raj", "amy", "bernadette"]:
        aList = []
        for name in ["sheldon", "leonard", "penny", "howard", "raj", "amy", "bernadette"]:
            aList.append(degreeDict.get(speaker, {}).get(name, 0))
        degreeList.append(aList)
    A = np.array(degreeList)


    # get eigenvector centrality
    G = nx.DiGraph(A)
    H = nx.relabel_nodes(G, {0:'sheldon', 1:'leonard', 2:'penny',3:'howard',4:'raj', 5:'amy', 6:"bernadette"})
    scores = nx.algorithms.centrality.eigenvector_centrality_numpy(H, weight='weight')
    print(scores)


def get_n_gram_distribution(data_dict, vocab_to_int, int_to_vocab, token_dict):
    "remove words first"
    # most popular bigrams in total
    # Most important bigrams for each character
    # bigram network?
    pass

def embedding_analysis(data_dict, vocab_to_int, int_to_vocab, token_dict):
    pass


def latent_semantic_indexing(data_dict, vocab_to_int, int_to_vocab, token_dict):
    pass



if __name__ == "__main__":

    data_dict = helper.load_data("data.pkl")
    int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

    # raw_stats(data_dict, vocab_to_int, int_to_vocab, token_dict)
    # degree_centrality(data_dict, vocab_to_int, int_to_vocab, token_dict)
