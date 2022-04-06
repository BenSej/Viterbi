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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""

import math

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    firstTagFreq = {}
    initial = {}
    transitionCounts = {}
    totalTagCounts = {}
    transition = {}
    emission = {}
    emissionCounts = {}
    words = {}
    wordTagMap = {}
    tagHapaxCount = {}
    tagHapax = {}

    k = .00001

    for sentence in train:
        firstTag = sentence[1][1]
        firstTagFreq[firstTag] = firstTagFreq.get(firstTag, 0) + 1
        for i in range(1, len(sentence) - 1):
            word, tag = sentence[i]

            if tag not in emissionCounts:
                emissionCounts[tag] = {}
            if tag not in transitionCounts:
                transitionCounts[tag] = {}

            if i >= 2:
                prevWord, prevTag = sentence[i - 1]
                transitionCounts[prevTag][tag] = transitionCounts[prevTag].get(tag, 0) + 1

            totalTagCounts[tag] = totalTagCounts.get(tag, 0) + 1
            emissionCounts[tag][word] = emissionCounts[tag].get(word, 0) + 1
            words[word] = words.get(word, 0) + 1
            wordTagMap[word] = tag
        break

    N = len(totalTagCounts)
    V = len(words)
    numSentences = len(train)
    tags = list(totalTagCounts.keys())

    for tag in tags:
        count = firstTagFreq.get(tag, 0) + k
        initial[tag] = count / (numSentences + k * N)

    for tag in tags:
        transition[tag] = {}
        for t in tags:
            count = transitionCounts[tag].get(t, 0)
            transition[tag][t] = (count + k) / (totalTagCounts[tag] + k * N)

    for tag in tags:
        emission[tag] = {}
        for word in emissionCounts[tag]:
            count = emissionCounts[tag][word]
            emission[tag][word] = (count + k) / (totalTagCounts[tag] + k * (V + 1))

    # extract hapax words
    hapaxWords = {}
    for word in words:
        if words[word] == 1:
            tag = wordTagMap[word]
            hapaxWords[word] = tag

    # count the hapax words for each tag
    for tag in tags:
        for word in hapaxWords:
            hapaxTag = wordTagMap[word]
            if hapaxTag == tag:
                tagHapaxCount[tag] = tagHapaxCount.get(tag, 0) + 1

    # compute tag hapax probabilities
    for tag in tags:
        count = tagHapaxCount.get(tag, k)
        total = sum(tagHapaxCount.values())
        probability = count / total
        tagHapax[tag] = probability

    sentences = []
    for sentence in test:
        prediction = predict(sentence, initial, transition, emission, words, k, totalTagCounts, N, V, numSentences, tagHapax)
        sentences.append(prediction)
    return sentences


def predict(sentence, initial, transition, emission, words, k, totalTagCounts, N, V, numSentences, tagHapax):
    tags = list(totalTagCounts.keys())
    lattice = {}
    backpointer = {}
    prevWord = None
    prevTag = None
    firstWord = True
    prediction = []
    index = 0
    for word in sentence:
        if word == "START" or word == "END":
            continue

        backpointer[(word, index)] = {}
        lattice[(word, index)] = {}

        if firstWord is True:
            for tag in tags:
                hapaxScaling = tagHapax[tag]
                piSmooth = k / (numSentences + k * N)
                bSmooth = k / (totalTagCounts[tag] + k * (V + 1)) * hapaxScaling
                probability = math.log(initial.get(tag, piSmooth)) + math.log(emission[tag].get(word, bSmooth))
                lattice[(word, index)][tag] = probability
                backpointer[(word, index)][tag] = None
            firstWord = False

        else:
            for tag in tags:
                hapaxScaling = tagHapax[tag]
                bSmooth = k / (totalTagCounts[tag] + k * (V + 1)) * hapaxScaling
                emissionProb = emission[tag].get(word, bSmooth)
                maxProb = -math.inf
                maxTag = ""
                for prevTag in tags:
                    aSmooth = k / (totalTagCounts[prevTag] + k * N)
                    currentProb = math.log(emissionProb) + math.log(transition[prevTag].get(tag, aSmooth)) + lattice[
                        (prevWord, index - 1)].get(prevTag, 0)
                    if currentProb > maxProb:
                        maxProb = currentProb
                        maxTag = prevTag
                lattice[(word, index)][tag] = maxProb
                backpointer[(word, index)][tag] = maxTag
        index += 1
        prevWord = word

    i = len(sentence) - 2
    word = sentence[i]
    tag = max(lattice[(word, i - 1)], key=lattice[(word, i - 1)].get)
    while i >= 1:
        word = sentence[i]
        prediction.append((word, tag))
        tag = backpointer[(word, i - 1)][tag]
        i -= 1
    prediction.append(("START", "START"))
    prediction.reverse()
    prediction.append(("END", "END"))
    return prediction