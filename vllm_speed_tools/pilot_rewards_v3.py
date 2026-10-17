"""Versioned pilot rewards; no changes to archived screening grades."""
import re
from fractions import Fraction
from math_verification import extract_final_answer, verify_final_answer, completion_text
from closure_rewards import complete_format

VERSION = 'sft-v3-pilot-numeric-units-mathverify-v2'
NUMBER = r'[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)'
UNITS = {'dollar','dollars','cent','cents','page','pages','hour','hours','minute','minutes',
 'second','seconds','day','days','week','weeks','month','months','year','years','mile','miles',
 'meter','meters','metre','metres','kilometer','kilometers','foot','feet','inch','inches',
 'pound','pounds','kg','gram','grams','gallon','gallons','liter','liters','litre','litres','ml',
 'student','students','person','people','child','children','book','books','ticket','tickets',
 'apple','apples','cookie','cookies','box','boxes','bag','bags','coin','coins','file','files',
 'hat','hats','hardhat','hardhats','driveway','driveways','dvd','dvds','item','items','unit','units',
 'shirt','shirts','ball','balls','bottle','bottles','cup','cups','piece','pieces','percent','%'}

def numeric_answer(value):
    s=str(value).strip()
    if s.startswith('$$') and s.endswith('$$'): s=s[2:-2].strip()
    elif s.startswith('$') and s.endswith('$'): s=s[1:-1].strip()
    s=re.sub(r'^[$£€]\s*','',s)
    s=re.sub(r'\\(?:text|mathrm)\{([^{}]+)\}',r'\1',s).strip()
    s=s.removesuffix('.').strip()
    s=re.sub(r'\\(?:dfrac|tfrac|frac)\{('+NUMBER+r')\}\{('+NUMBER+r')\}',r'\1/\2',s)
    m=re.fullmatch('('+NUMBER+r'(?:\s*/\s*'+NUMBER+r')?)\s*(.*)',s)
    if not m: return None
    suffix=m[2].strip().lower();scale=1
    for word,factor in [('thousand',1000),('million',1000000),('billion',1000000000)]:
        if suffix==word or suffix.startswith(word+' '):
            scale=factor;suffix=suffix[len(word):].strip();break
    if suffix and suffix not in UNITS: return None
    try:
        parts=m[1].replace(',','').replace(' ','').split('/')
        n=Fraction(parts[0]) if len(parts)==1 else Fraction(parts[0])/Fraction(parts[1])
        # A stated percent is not the same quantity as a bare proportion.
        return n*scale, ('percent' if suffix in {'percent','%'} else 'scalar')
    except (ValueError,ZeroDivisionError): return None

def grade(completion,gold,source):
    completion=completion_text(completion)
    pred,origin,error=extract_final_answer(completion)
    # Boxed fallback must also be unique in this experiment.
    if origin=='boxed' and len(re.findall(r'\\boxed\s*\{',completion))!=1:
        error='multiple_boxed_answers'
    if error: return dict(correct=False,status='extraction_failed',prediction=pred,error=error)
    if source=='gsm8k':
        a,b=numeric_answer(pred),numeric_answer(gold)
        if a is not None and b is not None:
            return dict(correct=a==b,status='correct' if a==b else 'incorrect',prediction=pred,error=None)
    return verify_final_answer(completion,gold)

def format_ok(completion,ended):
    completion=completion_text(completion)
    return bool(ended and complete_format(completion)
                and '<start_working_out>' not in completion)

def reward_pair(completion,gold,source,ended,weight):
    verdict=grade(completion,gold,source)
    return float(verdict['correct']),weight*float(format_ok(completion,ended)),verdict
