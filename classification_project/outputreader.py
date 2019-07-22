import csv
file = open("output5000.csv", "r")
lines = csv.reader(file,delimiter=',', quotechar='"') #lines is a list

incorrect = 0

for line in lines:
    if line[3] != line[4]:
        incorrect += 1


print(incorrect/5000)


"""
    real    fake
real

fake

"""
