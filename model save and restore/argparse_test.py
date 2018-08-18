import argparse

parser = argparse.ArgumentParser(prog="demo", description="A demo program", epilog='The end od usage')

parser.add_argument("name")
parser.add_argument("-a", '--age',action="store", type=int, required=True)
parser.add_argument('-s','--status',choices=['alpha', 'beta', 'released'], type=str, dest='mystatus')

args = parser.parse_args()
print(args)