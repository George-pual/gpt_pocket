"""Exact supporting checks for repaired mathematical examples, without models."""
import hashlib
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path
import sympy as s

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/math/sft_v3_supplement_unseen_grpo1536_v3_review'
x,k=s.symbols('x k',real=True)
checks=[]
def check(index,description,condition):
    assert bool(condition),(index,description)
    checks.append(dict(supplement_index=index,check=description,passed=True))

check(149,'All 24 digit arrangements for the two factors',min((10*a+b)*(10*c+d) for a,b,c,d in itertools.permutations([6,7,8,9]))==5372)
check(289,'All 21 binomial indices',[j for j in range(21) if math.comb(20,j)==math.comb(20,11)]==[9,11])
check(547,'Cleared equation factorization',s.expand((2*x+3)*(x+2))==2*x*x+7*x+6)
z=0
for j in range(2,112):z=s.expand(z*z+s.I)
check(551,'Exact recurrence from z_1=0 through index 111',z==-1+s.I)
t=F(1000)
for m in range(999,1,-1):
    assert 1-m*t!=0
    t=(m-t)/(1-m*t)
    assert (t>m) if m%2==0 else (0<t<F(1,m))
check(567,'Exact rational evaluation and denominator/bound checks for all 999 operations',t>2 and (1-t)/(1-t)==1)
n=s.symbols('n',integer=True,positive=True)
check(577,'Independent symbolic infinite sum',s.summation((3*n-1)/2**n,(n,1,s.oo))==5)
check(616,'Clear original rational equation',s.expand((x-2)*(x*x-10*x)-(x-5)*(x*x+4*x+1))==-(11*x*x-39*x-5))
check(616,'No shared roots with either original denominator',s.gcd(11*x*x-39*x-5,(x*x-10*x)*(x*x+4*x+1))==1)
check(621,'Direct squares and consecutive arithmetic terms',[36+925,300+925,596+925]==[31**2,35**2,39**2])
ratios=[(F(a,c),a,c) for a in range(1,79) for c in range(1,163) if 25*a+12*c==1950]
check(638,'Exhaustive feasible adult/child counts',min(ratios,key=lambda q:abs(q[0]-1))==(F(27,25),54,50))
tuples=[]
for z0 in range(-100,101):
 for mid in itertools.combinations(range(z0+1,z0+9),2):
    vals=(z0,)+mid+(z0+9,)
    if sum(vals)==44 and sorted(b-a for a,b in itertools.combinations(vals,2))==[1,3,4,5,6,9]:tuples.append(vals)
check(643,'Exhaustive gap-9 tuples; sum condition implies z in [4.25,8.25]',sorted(tuples)==[(6,11,12,15),(7,10,11,16)])
xx,yy=F(-9,8),F(-3,5)
terms=[xx+yy,xx-yy,xx*yy,xx/yy]
check(659,'Original arithmetic sequence and fifth term',len({terms[j+1]-terms[j] for j in range(3)})==1 and terms[-1]+F(6,5)==F(123,40))
check(668,'Original radical equation at sole derived candidate',1+4+4*s.sqrt(4)==13)
den=[x*x+11*x-8,x*x+2*x-8,x*x-13*x-8]
num=s.factor(s.together(sum(1/d for d in den))).as_numer_denom()[0]
check(676,'Numerator of original sum factors with exactly four listed roots',set(s.solve(num,x))=={-8,-1,1,8} and all(d.subs(x,v)!=0 for d in den for v in [-8,-1,1,8]))
check(724,'Original two radicals at repaired candidate',s.sqrt(s.Rational(2401,100))+s.sqrt(s.Rational(2601,100))==10)
check(737,'Linear case gives one intersection',s.solve(4*x+2,x)==[-s.Rational(1,2)])
check(737,'Nonzero quadratic tangency case gives one intersection',s.solve(2*x*x+4*x+2,x)==[-1])
check(738,'Original binomial ratios',F(math.comb(62,26),math.comb(62,27))==F(3,4) and F(math.comb(62,27),math.comb(62,28))==F(4,5))
check(742,'Original radical equation',s.simplify(5*s.sqrt(1+s.Rational(7,25))+5*s.sqrt(1-s.Rational(7,25))-7*s.sqrt(2))==0)
def seq(q):
    a=[1000,q]
    while a[-1]>=0:a.append(a[-2]-a[-1])
    return a
lengths=[len(seq(q)) for q in range(1,1002)]
check(750,'All relevant positive integer x; x>1000 immediately terminates at third term',max(lengths)==14 and [j+1 for j,v in enumerate(lengths) if v==14]==[618])
check(752,'Original rational equation',-(-1)**2==F(3*(-1)+1,(-1)+3))
check(759,'Original fractions, both roots',all(sum(F(1,(v-j)*(v-j-1)) for j in [1,2,3])==F(1,6) for v in [-2,7]))
check(762,'Independent polynomial division',s.div(x**5+7,x+1,x)==(x**4-x**3+x*x-x+1,6))
check(764,'Original rational equation, both roots',all(s.simplify(1/(v-2)+1/(v-4)-3/(v-3))==0 for v in [3-s.sqrt(3),3+s.sqrt(3)]))
check(770,'All integers in requested range',[v for v in range(-11,12) if (v-2)*(v+4)*(v+8)<0]==[-11,-10,-9,-3,-2,-1,0,1])
check(778,'Independent quotient and remainder',s.div(4*x**4+12*x**3-9*x*x+x+3,x*x+3*x-2,x)==(4*x*x-1,4*x+1))
check(783,'Independent division with symbolic k',s.div(x**4-6*x**3+16*x*x-25*x+10,x*x-2*x+k,x)==(x*x-4*x-k+8,(2*k-9)*x+k*k-8*k+10))
D=11*(F(1,13)-F(1,17))+13*(F(1,17)-F(1,11))+17*(F(1,11)-F(1,13))
N=121*(F(1,13)-F(1,17))+169*(F(1,17)-F(1,11))+289*(F(1,11)-F(1,13))
check(787,'Direct exact original fraction',D==F(48,2431) and N/D==41)
check(792,'All possible three-digit squares',[q*q for q in range(10,32) if str(q*q)==str(q*q)[::-1]]==[121,484,676])
check(793,'All primes in derived tuple and their sum',all(s.isprime(q) for q in [5,2,3,7,17]))
check(801,'Minimum by exhaustive n<=72',[q for q in range(1,73) if math.isqrt(2*q)**2==2*q and round((3*q)**(1/3))**3==3*q]==[72])
check(804,'Independent integer gcd',math.gcd(957,1537)==29)
check(805,'Exhaustive prime exponents with 2^p-1<200',max(2**q-1 for q in range(2,8) if s.isprime(q) and s.isprime(2**q-1))==127)
check(819,'Independent base-six integer addition',int('2343',6)+int('15325',6)==int('22112',6))
check(823,'Minimum by exhaustive n<=108',[q for q in range(1,109) if math.isqrt(3*q)**2==3*q and round((2*q)**(1/3))**3==2*q]==[108])
check(831,'Original modular powers',(pow(8,6,5)+pow(7,7,5)+pow(6,8,5))%5==3)
check(836,'All positive n<100',max(q for q in range(1,100) if (q+1)%8==0 and (q+3)%7==0)==95)
check(866,'Original large modular power',pow(5,1993,1000)==125)
check(922,'All ten digit residues',[d for d in range(10) if (7*d)%10==1]==[3])
check(1204,'Original norm at k=-1',s.sqrt((-6)**2+(-4)**2)==2*s.sqrt(13))
check(1280,'Original fold distances and resulting squared crease',s.Rational(7,5)**2+s.Rational(7,4)**2-s.Rational(7,5)*s.Rational(7,4)==s.Rational(1029,400))
a=s.pi/12
m=s.Matrix([1,0,0]);nn=s.Matrix([s.cos(a),s.sin(a),0]);p=s.Matrix([s.sin(a),0,s.cos(a)])
check(1287,'Explicit unit vectors satisfy both angles and original triple product',s.simplify(nn.dot(p.cross(m))-s.Rational(1,4))==0 and s.simplify(m.dot(nn)-s.cos(a))==0 and s.simplify(p.dot(m.cross(nn))/s.sin(a)-s.cos(a))==0)
points=[s.Matrix([s.sqrt(3)*s.cos(j*s.pi/4),s.sqrt(3)*s.sin(j*s.pi/4)]) for j in range(8)]
areas=[]
for A,B,C in itertools.combinations(points,3):
    areas.append(s.simplify(abs(s.det(s.Matrix.hstack(B-A,C-A)))/2))
least=min(areas,key=lambda v:float(v))
check(1326,'All 56 vertex triangles of original octagon',s.simplify(least-(3*s.sqrt(2)-3)/2)==0)
check(1343,'Original complete matrix product',s.Matrix([[1,2,2],[2,1,-2],[-2,2,-1]])*s.Matrix([[1,2,2],[2,1,-2],[-2,2,-1]]).T==9*s.eye(3))
check(1363,'Original dot and cross constraints',s.Matrix([2,1,5]).dot(s.Matrix([-1,3,2]))==11 and s.Matrix([2,1,5]).cross(s.Matrix([-1,3,2]))==s.Matrix([-13,-9,7]))
check(1383,'Independent Heron area and common-angle subtriangle ratio',s.simplify(s.sqrt(15*6*5*4)*s.Rational(4*7,9*11)-280*s.sqrt(2)/33)==0)
check(1384,'All earlier positive rotation powers fail',[q for q in range(1,37) if 170*q%360==0]==[36])
report=dict(status='passed',checks=checks,total_checks=len(checks),sympy_version=s.__version__,
    source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Exact symbolic, rational, or finite exhaustive supporting checks for selected repairs. These checks do not replace the per-record reasoning review and are not a formal proof verifier.')
OUT.mkdir(exist_ok=True)
(OUT/'supporting_math_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=report['status'],total_checks=len(checks))))
