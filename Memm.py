import MemmInference
import Trainer


def only_learn(mode):
    trainer = Trainer.Trainer()
    if mode != 'Basic' or mode != 'Improved' or mode != 'Comp':
        print('INVALID VALUE! Please choose Basic\\Improved\\Comp')
        return
    trainer.train(mode)


def only_inference_test_file(mode):
    infer = MemmInference.MemmInference()
    if mode != 'Basic' or mode != 'Improved' or mode != 'Comp':
        print('INVALID VALUE! Please choose Basic\\Improved\\Comp')
        return
    infer.check_acq_for_file_with_tags(mode)


def only_inference_competition_file(mode):
    infer = MemmInference.MemmInference()
    infer.inference_comp(mode)


