import fastwer
import jiwer  
import timeit
import re
import string
import Levenshtein

import mywer

def clean_string(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.split()
    text = [word.lower() for word in text if word not in string.punctuation]
    return text

def test_wer():
    h = clean_string('Mathworks connection programs')
    r = clean_string('MathWorks Connections Program')
    print(h, r)
    print(len(h), len(r))
    import timeit

    char = False
    number = 100000
    res = ''

    res = timeit.timeit('mywer.wer1(r, h)', setup='import mywer', number=number, globals = locals()) 
    print(mywer.wer1(r, h, char_level=char), res)

    res = timeit.timeit('mywer.wer2(r, h)', setup='import mywer', number=number, globals = locals()) 
    print(mywer.wer2(r, h, char_level=char), res)

    res = timeit.timeit('jiwer.wer(r, h)', setup='import jiwer', number=number, globals = locals())
    print(jiwer.wer(r if not char else list(''.join(r)), h if not char else list(''.join(h))), res)

    res = timeit.timeit('fastwer.score(h, r)', setup='import fastwer', number=number, globals = locals())
    print(fastwer.score(h, r, char_level=char), res)

if __name__ == "__main__":
    test_wer()