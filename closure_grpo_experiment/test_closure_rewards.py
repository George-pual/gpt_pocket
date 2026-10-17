"""Critical reward boundary checks; no model or GPU required."""
from closure_rewards import make_end_reward, soft_penalty, complete_format
from math_verification import math_verify_reward


def main():
    good='Compute 1+1=2.<end_working_out><SOLUTION>2</SOLUTION>'
    bad='Compute 1+1=3.<end_working_out><SOLUTION>3</SOLUTION>'
    samples=[good,bad,good,'Compute 1+1=2.<end_working_out><SOLUTION>2',r'Thus, \boxed{2}',good+good]
    ids=[[10,151643],[10,151643],[10],[10,151643],[10],[10,151643]]
    assert math_verify_reward(samples,['2']*6)==[1.,0.,1.,0.,1.,0.]
    assert make_end_reward(151643,.1)(samples,['2']*6,ids)==[.1,0.,0.,0.,0.,0.]
    assert make_end_reward(151643,0)(samples,['2']*6,ids)==[0.]*6
    assert [soft_penalty(n) for n in [100,3382,3638,3894,4000]]==[0.,0.,-.25,-.5,-.5]
    assert 1.+soft_penalty(3894)==.5  # Correct but cut off retains correctness reward.
    assert complete_format(good) and not complete_format(good+'more')
    print('REWARD_TESTS_PASSED',flush=True)


if __name__=='__main__':main()
