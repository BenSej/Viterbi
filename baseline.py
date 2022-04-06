# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    # maps words to their wordTag dictionaries
    master = {}
    # maps tags to frequencies for a given word
    wordTagFreq = {}
    # keep track of tag frequencies in the training set
    tags = {}
    for sentence in train:
        for pair in sentence:
            word = pair[0]
            tag = pair[1]
            tags[tag] = tags.get(tag, 0) + 1

            if word in master:
                wordTags = master[word]
                if tag in wordTags:
                    wordTags[tag] += 1
                else:
                    wordTags[tag] = 1
            else:
                dict = {}
                dict[tag] = 1
                master[word] = dict

    mostFreqTag = max(tags, key=tags.get)
    tagsList = list(tags.keys())

    result = []
    for sentence in test:
        taggedSentence = []

        for word in sentence:
            if word not in master:
                taggedSentence.append((word, mostFreqTag))
            else:
                tagDict = master[word]
                pair = (word, max(tagDict, key=tagDict.get))
                taggedSentence.append(pair)

        result.append(taggedSentence)

    return result