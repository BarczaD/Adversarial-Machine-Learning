from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
from foolbox.attacks import LinfPGD, L2PGD, LinfFastGradientAttack, L2FastGradientAttack
from foolbox.models import TensorFlowModel
import numpy as np
import os


def batch_attack(imgs, labels, attack, fmodel, eps, batch_size):
    adv = []
    for i in range(int(np.ceil(imgs.shape[0] / batch_size))):
        x_adv, _, success = attack(fmodel, imgs[i * batch_size:(i + 1) * batch_size],
                                   criterion=labels[i * batch_size:(i + 1) * batch_size], epsilons=eps)
        adv.append(x_adv)
    return np.concatenate(adv, axis=0)


def main(params):
    print(params)
    if params.step_size == 0:
        step_size = (params.eps / params.steps) * 4 / 3
    else:
        step_size = params.step_size
    netname = params.fname
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # it is important to cast the dtype otherwise the attack will throw exception
    x_test = np.array(x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255., np.float32)
    y_test = np.array(y_test, np.int64)
    if netname.endswith(".h5"):
        model = keras.models.load_model(netname)
    else:
        raise Exception("Network filename format is not supported: {0}".format(netname))
    pred = model.predict(x_test)
    if str.lower(params.attack) == 'pgd':
        if str.lower(params.norm) in ['linf', 'l_inf', 'l inf']:
            if str.lower(params.random_restart) == 'true':
                attack = LinfPGD(abs_stepsize=step_size, steps=params.steps, random_start=True)
            elif str.lower(params.random_restart) == 'false':
                attack = LinfPGD(abs_stepsize=step_size, steps=params.steps, random_start=False)
            else:
                raise ValueError("Unknown '--random_restart' value.")
        elif str.lower(params.norm) in ['l2', 'l_2', 'l 2']:
            if str.lower(params.random_restart) == 'true':
                attack = L2PGD(abs_stepsize=step_size, steps=params.steps, random_start=True)
            elif str.lower(params.random_restart) == 'false':
                attack = L2PGD(abs_stepsize=step_size, steps=params.steps, random_start=False)
            else:
                raise ValueError("Unknown '--random_restart' value.")
        else:
            raise ValueError("Unknown '--norm' value.")
    elif str.lower(params.attack) == 'fgsm':
        if str.lower(params.norm) in ['linf', 'l_inf', 'l inf']:
            if str.lower(params.random_restart) == 'true':
                attack = LinfFastGradientAttack(random_start=True)
            elif str.lower(params.random_restart) == 'false':
                attack = LinfFastGradientAttack(random_start=False)
            else:
                raise ValueError("Unknown '--random_restart' value.")
        elif str.lower(params.norm) in ['l2', 'l_2', 'l 2']:
            if str.lower(params.random_restart) == 'true':
                attack = L2FastGradientAttack(random_start=True)
            elif str.lower(params.random_restart) == 'false':
                attack = L2FastGradientAttack(random_start=False)
            else:
                raise ValueError("Unknown '--random_restart' value.")
        else:
            raise ValueError("Unknown '--norm' value.")
    else:
        raise ValueError("Unknown '--attack' value.")
    if params.trials > 1:
        attack = attack.repeat(params.trials)
    fmodel = TensorFlowModel(model, bounds=(0, 1), device='/device:GPU:0')
    imgs = tf.convert_to_tensor(x_test)
    labs = tf.convert_to_tensor(y_test)
    acc = np.mean(np.argmax(pred, axis=1) == y_test)
    x_adv = batch_attack(imgs, labs, attack, fmodel, params.eps, params.batch_size)
    p_adv = model.predict(x_adv)
    a_acc = np.mean(np.argmax(p_adv, axis=1) == y_test)
    print(netname)
    print('acc:', acc)
    print('robust-acc:', a_acc)


if __name__ == '__main__':
    parser = ArgumentParser(description='Model evaluation')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--norm', type=str, default='linf')
    parser.add_argument('--attack', type=str, default='pgd')
    parser.add_argument('--step_size', type=int, default=0)
    parser.add_argument('--random_restart', type=str, default='true')
    FLAGS = parser.parse_args()
    np.random.seed(9)
    if FLAGS.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

        physical_devices = tf.config.list_physical_devices('GPU')
        print("Physical devices: " + str(physical_devices))
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main(FLAGS)
