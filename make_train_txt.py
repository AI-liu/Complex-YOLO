
file = open("train.txt","w")


for i in range(6000):
    file_i = str(i).zfill(6)
    file.write(file_i + "\n")

file.close()