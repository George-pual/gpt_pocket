"""Apply explicit reasoning repairs to a new review draft; frozen v2 stays intact."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from remaining_reasoning_repairs_extra import REWRITES as EXTRA_REWRITES, ANSWER_OVERRIDES

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'datasets/math/sft_v3_supplement_unseen_grpo1536_v2'
AUDIT=SRC/'remaining_reasoning_audit'
OUT=ROOT/'datasets/math/sft_v3_supplement_unseen_grpo1536_v3_review'

REWRITES={
523:r'''Let $r>0$ be the radius. The given relation is $\pi r^2+2\pi r=80\pi$, so $r^2+2r-80=(r-8)(r+10)=0$. The two algebraic roots are 8 and $-10$; only the positive one can be a radius. Thus $r=8$.''',
547:r'''The original denominator is $(x+1)(x+2)$, so $x\ne-1,-2$. On this domain, canceling $x+2$ gives $x^2/(x+1)+x=-6$. Multiplying by the nonzero $x+1$ yields $2x^2+7x+6=(2x+3)(x+2)=0$. The candidate $-2$ is excluded, while $x=-3/2$ is allowed and satisfies the original equation. Hence it is the only solution.''',
548:r'''Let $A=1/[(1-x)(1-y)(1-z)]$ and $B=1/[(1+x)(1+y)(1+z)]$. The hypotheses make both terms positive. Their product is $AB=1/[(1-x^2)(1-y^2)(1-z^2)]\ge1$, because each denominator factor is in $(0,1]$. Therefore AM-GM gives $A+B\ge2\sqrt{AB}\ge2$. At $x=y=z=0$, both terms equal 1, so the minimum is 2.''',
551:r'''Compute $z_2=i$, $z_3=-1+i$, $z_4=(-1+i)^2+i=-i$, and $z_5=(-i)^2+i=-1+i=z_3$. Since the recurrence depends only on its previous term, the sequence repeats with period 2 from index 3: odd indices at least 3 have value $-1+i$, and even indices at least 4 have value $-i$. Thus $z_{111}=-1+i$ and its distance from the origin is $\sqrt{(-1)^2+1^2}=\sqrt2$.''',
552:r'''For $(a,b)=(2,1)$, the sum $a+b=3$, so use the first branch: $f(2,1)=(2\cdot1-2+2)/(2\cdot2)=1/2$. For $(2,4)$ the sum is 6, so use the second branch: $f(2,4)=(2\cdot4-4-2)/(-2\cdot4)=-1/4$. Hence $f(2,1)+f(2,4)=1/2-1/4=1/4$.''',
559:r'''The divisor has degree 5. Polynomial division gives a remainder that is either zero or has degree strictly less than 5. A nonzero polynomial has nonnegative integer degree, so the possible degrees are $0,1,2,3,4$, and each occurs by dividing the polynomial $x^k$ for the corresponding $k<5$. The zero remainder has no degree under the usual convention and adds no integer to the requested list.''',
560:r'''The domain excludes $x=-10,-9,-7,-6$. Combining each side gives $(2x+16)/[(x+9)(x+7)]=(2x+16)/[(x+10)(x+6)]$. On the stated domain, cross-multiplication is valid and gives $2(x+8)[(x+10)(x+6)-(x+9)(x+7)]=-6(x+8)=0$. Thus $x=-8$, which is allowed. Substituting gives $1+(-1)=1/2+(-1/2)=0$, confirming the solution.''',
562:r'''Write the first three increasing positive integers as $a,a+d,a+2d$, with $a,d>0$ integers. The fourth is $a+30$. Since the last three form a geometric progression, $(a+d)(a+30)=(a+2d)^2$, giving $a(30-3d)=4d^2-30d$. For $d=10$ this equation is impossible. Otherwise $a=2d(2d-15)/[3(10-d)]$. Its positivity with $d>0$ requires $15/2<d<10$, so $d=8$ or 9. The first gives $a=8/3$, not an integer; the second gives $a=18$. The resulting sequence $18,27,36,48$ is increasing, has the required arithmetic and geometric progressions, and sums to 129.''',
563:r'''An even starting value immediately decreases, so cannot work. If $a_1=4k+1$ with $k\ge0$, the next terms are $12k+4,6k+2,3k+1$. The fourth satisfies $3k+1\le4k+1=a_1$, including equality at $k=0$, so this case also fails the strict inequalities. If $a_1=4k+3$, the next terms are $12k+10,6k+5,18k+16$, all strictly greater than $4k+3$ for $k\ge0$. Thus exactly the integers congruent to 3 modulo 4 qualify. They are $3,7,\ldots,2007$, numbering $(2007-3)/4+1=502$.''',
567:r'''Define $T_{1000}=1000$ and $T_m=(m-T_{m+1})/(1-mT_{m+1})$ for $m=999,\ldots,2$. For $2\le m\le999$, we prove by downward induction that $T_m>m$ for even $m$, while $0<T_m<1/m$ for odd $m$. The base is $T_{999}=(1000-999)/(999\cdot1000-1)\in(0,1/999)$. For even $m\le998$, the following odd tail satisfies $0<t=T_{m+1}<1/(m+1)<1/m$, so the denominator is positive and $T_m-m=t(m^2-1)/(1-mt)>0$. For odd $m\le997$, the following even tail satisfies $t=T_{m+1}>m+1$, giving $T_m=(t-m)/(mt-1)\in(0,1/m)$, since $m(t-m)<mt-1$. These bounds also prove every intermediate denominator is nonzero. In particular $T_2>2$, so the final operation is defined and $1*T_2=(1-T_2)/(1-T_2)=1$.''',
15:r'''Subtracting 9 gives $x^2-5x-4=0$. If its roots are $r,s$, then $(x-r)(x-s)=x^2-(r+s)x+rs$. Comparing the coefficient of $x$ gives $r+s=5$. The discriminant is $25+16=41>0$, so both roots are real.''',
57:r'''Write both sides with base 3: $(3^3)^8=(3^2)^q$, so $3^{24}=3^{2q}$. Since $3^x$ is injective, $24=2q$, giving $q=12$.''',
131:r'''The rational expression is undefined exactly when its denominator is zero. Factoring the quadratic gives $8x^2-65x+8=(8x-1)(x-8)$. Thus the excluded values are $1/8$ and $8$, of which the larger is $8$.''',
134:r'''The non-white bows account for $1/5+1/2+1/10=4/5$ of the bag. Thus the 30 white bows are $1/5$ of the total, so there are 150 bows. The green bows are $150/10=15$.''',
149:r'''For any fixed pair of digits $u<v$, placing $u$ in the tens position makes the smaller two-digit number, since $10u+v<10v+u$. Both factors are positive, so reversing a sorted pair cannot decrease their product. There are exactly three ways to partition the four digits into two unordered pairs. Their smallest possible products are $67\cdot89=5963$, $68\cdot79=5372$, and $69\cdot78=5382$. Comparing all three gives the minimum $5372$.''',
170:r'''Let $f,s$ be the pounds of flour and sugar. The constraints are $6+s/2\le f\le2s$. Hence $6+s/2\le2s$, which gives $s\ge4$. This lower bound is attained by $s=4,f=8$, which satisfies both inequalities. Thus the least possible amount of sugar is 4 pounds.''',
178:r'''For every real $x$, $N(O(x))=2\sqrt{x^2}=2|x|$. In this expression all successive inputs to $N\circ O$ are positive, so each application doubles its input. Applying the composition three times gives $3\mapsto6\mapsto12\mapsto24$.''',
181:r'''The slope of $AB$ is $(-4-0)/(0-(-4))=-1$. Parallelism requires $(k-8)/(14-0)=-1$, so $k-8=-14$ and $k=-6$. Thus $Y=(14,-6)$.''',
189:r'''Factor $-b^2+7b-10=(b-2)(5-b)$. The product is zero at 2 and 5, positive for $2<b<5$, and negative for $b<2$ or $b>5$. The inequality therefore holds exactly on $[2,5]$, and its greatest solution is 5.''',
220:r'''Distribute and combine like terms: $(x+3)(x-8)=x(x+3)-8(x+3)=x^2+3x-8x-24=x^2-5x-24$.''',
231:r'''Factor $a^2-10a+21=(a-3)(a-7)$. It is zero at 3 and 7, negative between them, and positive for $a<3$ or $a>7$. The solution set is $[3,7]$, whose greatest value is 7.''',
260:r'''The original mixture contains $0.30\cdot30=9$ ounces of acid. Adding $w$ ounces of pure water leaves this amount unchanged and increases the total volume to $30+w$ ounces. Thus the acid concentration is $9/(30+w)$. Setting it equal to $0.20=1/5$ gives $30+w=45$, so $w=15$ ounces.''',
267:r'''Let the two numbers be $x,y$. Their difference has magnitude 9, so $(x-y)^2=81$. Since $x^2+y^2=153$, the identity $(x-y)^2=x^2+y^2-2xy$ gives $81=153-2xy$. Hence $xy=36$. This argument includes both possible pairs $(12,3)$ and $(-3,-12)$ when $x-y=9$.''',
285:r'''The first character has 26 letter choices and the last has 10 digit choices. These two characters differ because one is a letter and the other a digit. To produce a repeated character, the middle character must equal exactly one of them, giving two choices. Thus there are $26\cdot10\cdot2=520$ valid plates.''',
289:r'''Pascal's identity gives $\binom{21}{11}=\binom{20}{10}+\binom{20}{11}$, so the equation is $\binom{20}{n}=\binom{20}{11}=\binom{20}{9}$. For $0\le k\le9$, the ratio $\binom{20}{k+1}/\binom{20}{k}=(20-k)/(k+1)>1$. Together with $\binom{20}{k}=\binom{20}{20-k}$, this proves the only possible indices are 9 and 11. Their sum is 20.''',
319:r'''Rotation of the equilateral triangle by $120^\circ$ preserves uniform area probability and cyclically permutes the three areas $[ABP],[BCP],[CAP]$. Consequently each has the same probability of being the unique largest. An equality between two areas occurs only on their equal-distance bisector, a line segment with area zero. Thus, with probability one, exactly one of the three areas is largest. The three equal probabilities sum to 1, so the requested probability is $1/3$.''',
344:r'''Each plane determined by the points contains at least one non-collinear triple, so there are at most $\binom{12}{3}=220$ such planes. This bound is attained by choosing 12 points in general position, with no three collinear and no four coplanar. Such points can be chosen successively by avoiding the finitely many lines and planes determined by earlier points. Then every triple determines a plane, and different triples give different planes, since otherwise at least four points would be coplanar. Hence the maximum is 220.''',
352:r'''Initially there are $5\cdot3\cdot4=60$ plates. The six ways to allocate the two new letters among the three sets give size triples $(7,3,4),(5,5,4),(5,3,6),(6,4,4),(6,3,5),(5,4,5)$. Their products are respectively $84,100,90,96,90,100$. Thus the maximum total is 100 plates, giving $100-60=40$ additional plates.''',
367:r'''Choose the one person left out in 7 ways. For the six selected people, the number of circular seatings up to rotation is $6!/6=5!$. Therefore there are $7\cdot5!=840$ ways to choose and seat six of the seven people.''',
467:r'''Tripling every linear dimension multiplies volume by $3^3=27$, regardless of the container's shape: each small volume element has its three perpendicular lengths multiplied by 3. The new capacity is therefore $2\cdot27=54$ gallons.''',
486:r'''Let $L$ be the longest side. By the strict triangle inequality, $L<20-L$, so $L<10$. Since $L$ is an integer, $L\le9$. The three distinct side lengths $9,8,3$ sum to 20 and satisfy $9<8+3$, so the bound 9 is attained.''',
502:r'''Each right triangle has a $60^\circ$ angle at $E$, so its short leg is half its hypotenuse and its long leg is $\sqrt3/2$ times its hypotenuse. In $ABE$, $AE=24$, giving $BE=12$ and $AB=12\sqrt3$. In $BCE$, the hypotenuse is $BE=12$, giving $CE=6$ and $BC=6\sqrt3$. In $CDE$, the hypotenuse is $CE=6$, giving $DE=3$ and $CD=3\sqrt3$. The three consecutive angles at $E$ sum to $180^\circ$, so $A,E,D$ are collinear with $E$ between $A,D$. Hence $AD=AE+ED=27$. The perimeter is $AB+BC+CD+AD=27+21\sqrt3$.''',
507:r'''Reduce the angle modulo $360^\circ$: $-3645^\circ\equiv-45^\circ\equiv315^\circ$. On the unit circle, $P=(\sqrt2/2,-\sqrt2/2)$ at this angle. If $D$ is its perpendicular projection onto the horizontal axis and $O$ is the origin, the leg lengths are $DO=DP=\sqrt2/2$, while the hypotenuse $OP=1$. Therefore $\tan(-3645^\circ)=(-\sqrt2/2)/(\sqrt2/2)=-1$.''',
}
REWRITES.update(EXTRA_REWRITES)

def main():
    rows=[json.loads(x) for x in (SRC/'supplement_records.jsonl').read_text('utf-8').splitlines()]
    notes=[json.loads(x) for x in (AUDIT/'reviews.jsonl').read_text('utf-8').splitlines()]
    by_index={x['supplement_index']:x for x in notes}
    repairs=[]
    for row in rows:
        i=row['supplement_index']
        if i in by_index:
            note=by_index[i]
            if note['verdict']=='needs_fix':
                if i not in REWRITES: continue
                old=row['solution'];old_answer=row['answer']
                row['answer']=ANSWER_OVERRIDES.get(i,old_answer)
                row['solution']=REWRITES[i]+'\nFinal answer: \\boxed{'+row['answer']+'}'
                row['messages'][-1]['content']=row['solution']
                repairs.append(dict(supplement_index=i,row_id=row['row_id'],reason=note['reasoning_check'],
                    previous_answer=old_answer,answer=row['answer'],before=old,after=row['solution']))
                row['remaining_revision']=dict(previous_answer=old_answer,answer=row['answer'],
                    previous_solution_sha256=hashlib.sha256(old.encode()).hexdigest(),
                    reason=note['reasoning_check'])
            row['remaining_manual_review']=note
            row['mathematical_steps_independently_reviewed']=note['verdict']=='pass' or i in REWRITES
    OUT.mkdir(exist_ok=True)
    def write(name,value):
        (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in value),encoding='utf-8')
    write('supplement_records_review_draft.jsonl',rows)
    write('reasoning_repairs.jsonl',repairs)
    summary=dict(source_sha256=hashlib.sha256((SRC/'supplement_records.jsonl').read_bytes()).hexdigest(),
        total=len(rows),previous_reviewed=80,new_reviewed=len(notes),new_repaired=len(repairs),
        verdicts=dict(Counter(n['verdict'] for n in notes)),
        remaining=1310-len(notes),unapplied_repairs=[n['supplement_index'] for n in notes
            if n['verdict']=='needs_fix' and n['supplement_index'] not in REWRITES],
        status='IN_PROGRESS' if len(notes)<1310 else 'REVIEW_COMPLETE',
        training_started=False,description='Explicit per-item mathematical review. Source and original answers preserved. Draft is not a finalized training export.')
    (OUT/'review_progress.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
