import unittest
from lmm.embodiments.alphabot2.replay_events import replay_events


class ReplayEvents(unittest.TestCase):
    def test_event_order_and_state_readback(self):
        seen=[]
        n=replay_events([{'kind':'contacts'}, {'kind':'step','action':[0]*18},
                         {'kind':'contacts'}, {'kind':'step','action':[1]*18}],
                        lambda a:seen.append(('step',float(a[0]))),
                        lambda:seen.append('contacts'),lambda:seen.append('state'))
        self.assertEqual(n,2)
        self.assertEqual(seen,['contacts',('step',0.),'state','contacts',('step',1.),'state'])

    def test_unknown_or_invalid_events_fail_closed(self):
        for event in ({'kind':'other'}, {'kind':'step','action':[0]*17},
                      {'kind':'step','action':[float('nan')]*18}):
            with self.assertRaises(ValueError):
                replay_events([event],lambda _:self.fail('invalid action executed'),lambda:None)

if __name__=='__main__': unittest.main()
