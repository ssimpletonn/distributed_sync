import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model')
parser.add_argument('-d','--dataset')

args = parser.parse_args()

if(args.dataset == 'imagenet'):
    print('imagenet')
elif(args.dataset=='cifar10'):
    print('cifa')
else:
    print('nothing')