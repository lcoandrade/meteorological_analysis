import re

file = "stations/pomasqui.csv"
file2 = "stations/pomasqui1.csv"

with open(file, "r") as f:
    lines = f.readlines()

with open(file2, "w") as f:
    for line in lines:
        f.write(re.sub(";\n$", "\n", line))  # Substitui ';' por '\n' no final da linha
