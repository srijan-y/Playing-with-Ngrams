"""Uses corpora (a collection of texts) to automatically generate sentences.

To generate a sequence of words, the program calculates the probability
that a word follows specific word/phrase. It randomly picks the next word
based on a probability distribution -- a series of ranges that each correspond
to likelihoods of occurence.

The size (number of words) of the phrase to use for generation is N;
a statistical model created from the likelihood of N-sized phrases is called
an N-gram model.

Usage:
    The program can be called via a command-line interface.
    It takes 3+ arguments:
        arg1 is the N-value, used for computing ngrams
        arg2 is M, the number of sentences to generate
        args 3+ are the texts to pull words from

    Accepts file names from Project Gutenberg @ https://www.gutenberg.org

    Command format:
        python ngrams.py <N-value> <# of sentences to generate> <text1> [<text2>, <text3>, ...]

Example Input/Output:
    $ py ngrams.py 3 10 bible-kjv.txt
    > This program generates random sentences based on an Ngram model.
    > Command line settings: ngram.py 3 10

    > 5:23 drink no water.

    > 132:15 i will lay thee an ark of god by reason of his disciples over the which they had
    > committed abomination before me to house.

    > 34:10 now when daniel knew that he hath judged the great depths.

    > 31:42 and of every debt.

    > and sent and sanctified eleazar his son reigned in his house.

    > 7:52 one kid of the thessalonians which is preferred before me.

    > 12:37 and at jerusalem.

    > 76:2 in salem also is vanity.

    > for mine iniquities and for the lord god.

    103:22 bless the lord.

Algorithm:
    starts in main():
        - greets the user
        - prints the entered arguments
        - calls train_model()

    train_model() creates the ngram model:
        - pulls the text Project Gutenberg's collection
        - tokenizes the text into sentences:
            - adds <start> and <end> tokens
        - creates the bigram conditional probability distribution
        - creates the ngram conditional probability distribution
        - returns a tuple in the form of (bigram_cfd, ngram_cfd)
        - control goes back to main()

    main() enters a loop:
        - calls generate_sentence()

    generate sentences():
        - starts by picking a word that begins the sentence
            - uses the <start> token
        - if N > 2:
            - need to generate until there are N words
            - bigram model for generation
        - switch to N-gram model when there are N words
        - generation ends when punctuation is added
        - returns the sentence

    back to main():
        - if sentence is valid:
            - prints each sentence as it is returned
            - increments sentence count
        - ends when count == M, otherwise loop continues

:author name: Rav Singh, Srijan Yenumula
:class: IT-499-002
:date: 19-Feb-2018
"""
from sys import argv

from nltk import (ConditionalFreqDist, ConditionalProbDist, LaplaceProbDist, ngrams,
                  sent_tokenize, word_tokenize)
from nltk.corpus import gutenberg

# N-value determines N-gram size
N_VAL = int(argv[1])
N_MINUS1 = N_VAL - 1

# requested number of sentences
SENT_COUNT = int(argv[2])

# list of texts to pull from
CORPORA = argv[3:]


def train_model():
    """Create ngram model from Project Gutenberg texts"""
    text = ''
    for corpus in CORPORA:
        with open(corpus, 'r') as file_:
            text += file_.read().replace('\n', '')

    sents = sent_tokenize(text.lower())
    tokens = []
    # appends <start> and <end> tokens to each sentence
    for sent in sents:
        sent = 'START ' + sent + ' END'
        tokens += word_tokenize(sent)

    ngrams_ = tuple(ngrams(tokens, N_VAL))

    # bigram frequency distribution
    bi_cfdist = ConditionalFreqDist(
        (ngram[0], ngram[:2])
        for ngram in ngrams_
    )

    # bigram probability distribution
    bi_cpdist = ConditionalProbDist(bi_cfdist, LaplaceProbDist)

    # conditional frequency distribution
    cfdist = ConditionalFreqDist(
        (ngram[:N_MINUS1], ngram)
        for ngram in ngrams_
    )

    # conditional probability
    cpdist = ConditionalProbDist(cfdist, LaplaceProbDist)

    return bi_cpdist, cpdist


def generate_sentence(bi_cpdist, cpdist):
    """Randomly generate a sentence based on probability distributions

    :param bi_cpdist: bigram probability distribution\n
    :param cpdist: ngram probability distribution
    """
    try:
        sample = bi_cpdist['START'].generate()[1]
    except ValueError:
        return ''

    retval = sample + ' '

    # if N > 2, randomly generate words using a bigram model
    # continue until there are N words to use as a sample
    if N_VAL > 2:
        ctr = N_VAL
        while ctr > 2:
            word = bi_cpdist[sample].generate()[1]
            if word not in ',:;\'".!?':
                retval += word + ' '
                sample = word
                ctr -= 1
            if word in '.!?':
                return retval[:-1] + word

    sample = tuple(retval.split())
    word = ''
    while True:
        try:
            sample = cpdist[sample].generate()
        except ValueError:
            return ''

        word = sample[N_MINUS1]
        if word not in ',:;\'".!?':
            retval += word + ' '
            sample = sample[1:N_VAL]
        if word in '.!?':
            return retval[:-1] + word


def main():
    """Entry point for program"""
    greeting = 'This program generates random sentences based on an N-gram model.'
    print(
        greeting,
        f'Command line settings: ngram.py {N_VAL} {SENT_COUNT}',
        sep='\n',
        end='\n\n'
    )

    bi_cpdist, cpdist = train_model()
    ctr = 0
    while ctr < SENT_COUNT:
        sent = generate_sentence(bi_cpdist, cpdist)
        if len(sent.split()) > N_VAL:
            print(sent, end='\n\n')
            ctr += 1


if __name__ == '__main__':
    main()
