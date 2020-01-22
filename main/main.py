import sys
import json
import numpy as np

# Read .in date file
inFile = open(sys.argv[1])
data = np.loadtxt(inFile)
X = data[:,0:2]
xtra = np.transpose(np.ones(X.shape[0]))
X = np.c_[xtra, X] # Add column of 1's to left side of X
y = data[:,2]


# Calculate analytic weights
X_transpose = np.transpose(X)
w_analytic = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
#print(w_analytic)


# Interpret JSON
with open(sys.argv[2]) as f:
	jsonData = json.load(f)
learningRate = jsonData["learning rate"]
iterations = jsonData["num iter"]

# Calculate stochastic gradient descent weights
w_gd = np.random.rand(X.shape[1])

for iter in range(iterations):
	n = np.random.randint(0,y.shape[0])
	X_n = X[n, :].reshape(1, X.shape[1])
	y_n = y[n]
	y_guess = X_n.dot(w_gd)
	diff = y_n - y_guess
	w_gd = w_gd + learningRate*diff.dot(X_n)


# Create output file name
outFileName = sys.argv[1].split("."[0]) + ".out"

outFile = open(outFileName, "w")
outFile.write(w_analytic + "\n" + w_gd)
outFile.close()

print(w_analytic)
print(w_gd)