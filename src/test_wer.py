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

    char = True
    res = ''

    res = timeit.timeit('mywer.wer1(r, h)', setup='import mywer', number=10000, globals = locals()) 
    print(mywer.wer1(r, h, char_level=char), res)

    res = timeit.timeit('mywer.wer2(r, h)', setup='import mywer', number=10000, globals = locals()) 
    print(mywer.wer2(r, h, char_level=char), res)

    res = timeit.timeit('jiwer.wer(r, h)', setup='import jiwer', number=10000, globals = locals())
    print(jiwer.wer(r if not char else list(''.join(r)), h if not char else list(''.join(h))), res)

    res = timeit.timeit('fastwer.score(h, r)', setup='import fastwer', number=10000, globals = locals())
    print(fastwer.score(h, r, char_level=char), res)

test_wer()