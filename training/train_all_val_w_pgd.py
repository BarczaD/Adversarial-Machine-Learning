import os
import csv
import subprocess as sp
import time

'''def main():
    for steps_m in [40, 100]:
        for steps_s in [40, 100]:
            for eps_m in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for eps_s in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    run_command("fgsm", steps_m, "linf", eps_m, "True", steps_s, eps_s)'''

def main():
    for model in ['lpdcnna','samplecnn']:
        if model == 'lpdcnna':
            run_command('fgsm', 100, 'linf', 0.2, 'True', 100, 0.2, model)
        else:
            run_command('fgsm', 100, 'linf', 0.3, 'True', 100, 0.3, model)




def run_command(attack, steps, norm, eps, random_restart, val_steps, val_eps, model):
    proc = sp.Popen(f'python train_new_val_w_pgd.py --gpu 0 --attack {attack} --steps {steps} --eps {eps} --random_restart {random_restart} --val_steps {val_steps} --val_eps {val_eps} --batch_size 1000 --model {model}', shell=True, stdout=sp.PIPE)
    splitted = str(proc.communicate()[0]).split("\\")
    print(splitted)
    # Puts the acc's and robust-acc's value into the 'output' variable as a list.
    # IMPORTANT: different operating systems might give different outputs! This code was tested under Linux and might not work under Windows or Mac!
    # output = [splitted[2].split(':')[1], splitted[3].split(':')[1]]
    # Prints out the results on the console.
    # print(lines + os.linesep + 'Output: acc:' + output[0] + ', robust-acc:' + output[1] + os.linesep + lines)
    # Writes the output into the csv.
    # writer.writerow([attack, trials, eps, steps, norm, fname.split('/')[2], val_steps, val_eps, str(output[0]), str(output[1]), str(time.time() - runtime) + 's'])




if __name__ == '__main__':
    # Preparing the csv file, that will contain the results.
    with open('results_w_val_pgd.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['attack','trials', 'eps', 'steps', 'norm', 'fname', 'val_steps', 'val_eps' , 'acc', 'robust-acc', 'runtime'])

        main()
