import re

file = "../stations/estacion_desconocida1.csv"
file2 = "../stations/estacion_desconocida11.csv"

with open(file, "r") as f:
    lines = f.readlines()

with open(file2, "w") as f:
    for line in lines:
        f.write(re.sub(";\n$", "\n", line))  # Substitui ';' por '\n' no final da linha
