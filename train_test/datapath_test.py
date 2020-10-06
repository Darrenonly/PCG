from config.config import FEATPATH

fh = open(FEATPATH + 'train1.txt', 'r')
imgs = []
for line in fh:
    line = line.strip('\n')
    line = line.rstrip('\n')
    content = line.split(" ")
    if content[1] == '':
        imgs.append((content[2], 0))
    else:
        imgs.append((content[2], int(content[1])))
print(imgs)
