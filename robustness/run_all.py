import os
import csv
import subprocess as sp
import time

# For making console outputs more readable.
lines = '--------------------------------------------------------------------------------------------'

listOfFoldersAndFiles = os.listdir('best_final/')
fullpaths = map(lambda name: os.path.join('best_final/', name), os.listdir('best_final/'))
files = []



def main():
    _runtime = time.time()
    for method in ['_fgsm', '']:
        for steps in [40, 100]:
            for norm in ['linf', 'l2']:
                if norm == 'linf':
                    for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        for fname in files:
                            if method == '':
                                run_command('pgd', steps, norm, eps, './best_final/' + fname)
                            else:
                                run_command('fgsm', steps, norm, eps, './best_final/' + fname)
                else:
                    for eps in [1, 1.5, 2, 2.5]:
                        for fname in files:
                            if method == '':
                                run_command('pgd', steps, norm, eps, './best_final/' + fname)
                            else:
                                run_command('fgsm', steps, norm, eps, './best_final/' + fname)

    with open('runtime.txt', "w") as f:
        f.write('Total runtime of last execution: ' + str(time.time() - _runtime) + 's.')


# Checks robustness against FGSM attacks.
def fgsm_run(steps, norm, eps, fname):
    print(lines + os.linesep + 'Running: ' + 'python main_fgsm.py --gpu 0 --steps ' + str(steps) + ' --eps ' + str(eps) + ' --norm ' + str(norm) + ' --fname ' + str(
        fname) + os.linesep + lines)
    # Saves the main_fgsm.py output into the 'proc' variable (tuple).
    runtime = time.time()
    proc = sp.Popen(
        'python main_fgsm.py --gpu 0  --eps ' + str(
            eps) + ' --norm ' + str(norm) + ' --fname ' + str(fname), shell=True, stdout=sp.PIPE)
    splitted = str(proc.communicate()[0]).split("\\")
    # Puts the acc's and robust-acc's value into the 'output' variable as a list.
    # IMPORTANT: different operating systems might give different outputs! This code was tested under Linux and might not work under Windows or Mac!
    output = [splitted[2].split(':')[1], splitted[3].split(':')[1]]
    # Prints out the results on the console.
    print(lines + os.linesep + 'Output: acc:' + output[0] + ', robust-acc:' + output[1] + os.linesep + lines)
    # Writes the output into the csv.
    writer.writerow(['fgsm', '1', eps, steps, norm, fname.split('/')[2], str(output[0]), str(output[1]), str(time.time() - runtime)])


# Checks robustness against PGD attacks.

def pgd_run(steps, norm, eps, fname):
    for trials in [1, 10]:
        print(lines + os.linesep + 'Running: ' + 'python main.py --gpu 0 --trials ' + str(
            trials) + ' --steps ' + str(steps) + ' --eps ' + str(eps) + ' --norm ' + str(norm) + ' --fname ' + str(
            fname) + os.linesep + lines)
        # Saves the main.py output into the 'proc' variable (tuple).
        runtime = time.time()
        proc = sp.Popen(
            'python main.py --gpu 0 --trials ' + str(trials) + ' --steps ' + str(steps) + ' --eps ' + str(
                eps) + ' --norm ' + str(norm) + ' --fname ' + str(fname), shell=True, stdout=sp.PIPE)
        splitted = str(proc.communicate()[0]).split("\\")
        # Puts the acc's and robust-acc's values into the 'output' variable as a list.
        # IMPORTANT: different operating systems might give different outputs! This code was tested under Linux and might not work under Windows or Mac!
        output = [splitted[2].split(':')[1], splitted[3].split(':')[1]]
        # Prints out the results on the console.
        print(lines + os.linesep + 'Output: acc:' + output[0] + ', robust-acc:' + output[1] + os.linesep + lines)
        # Writes the output into the csv.
        writer.writerow(['pgd', trials, eps, steps, norm, fname.split('/')[2], str(output[0]), str(output[1]), str(time.time() - runtime) + 's'])


def run_command(attack, steps, norm, eps, fname):
    for trials in [1, 10]:
        print(lines + os.linesep + 'Running: ' + 'python main_new.py --gpu 0 --attack ' + attack  + ' --trials ' + str(
            trials) + ' --steps ' + str(steps) + ' --eps ' + str(eps) + ' --norm ' + str(norm) + ' --fname ' + str(
            fname) + os.linesep + lines)
        # Saves the main.py output into the 'proc' variable (tuple).
        runtime = time.time()
        proc = sp.Popen(
            'python main_new.py --gpu 0 --attack ' + attack + ' --trials ' + str(trials) + ' --steps ' + str(steps) + ' --eps ' + str(
                eps) + ' --norm ' + str(norm) + ' --fname ' + str(fname), shell=True, stdout=sp.PIPE)
        splitted = str(proc.communicate()[0]).split("\\")
        # print("proc.communicate(): "  + str(proc.communicate()[0]))
        # Puts the acc's and robust-acc's value into the 'output' variable as a list.
        # IMPORTANT: different operating systems might give different outputs! This code was tested under Linux and might not work under Windows or Mac!
        output = [splitted[3].split(':')[1], splitted[4].split(':')[1]]
        # Prints out the results on the console.
        print(lines + os.linesep + 'Output: acc:' + output[0] + ', robust-acc:' + output[1] + os.linesep + lines)
        # Writes the output into the csv.
        writer.writerow([attack, trials, eps, steps, norm, fname.split('/')[2], str(output[0]), str(output[1]), str(time.time() - runtime)])

if __name__ == '__main__':
    # Preparing the csv file, that will contain the results.
    with open('results_final.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['attack','trials', 'eps', 'steps', 'norm', 'fname', 'acc', 'robust-acc', 'runtime'])
        files = []
        for e in fullpaths:
            trippedTmp = str.split(e, "/")
            files.append(trippedTmp[len(trippedTmp) - 1])

        main()
