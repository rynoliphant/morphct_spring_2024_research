import os
import csv
import numpy as np
import cPickle as pickle
import random as R
import matplotlib
import matplotlib.pyplot as plt
import helperFunctions
try:
    import mpl_toolkits.mplot3d.axes3d as p3
except ImportError:
    print "Could not import 3D plotting engine, calling the plotCarrier subroutine will result in an error!"

    
plottingSubroutines = True
elementaryCharge = 1.60217657E-19 # C
kB = 1.3806488E-23 # m^{2} kg s^{-2} K^{-1}
hbar = 1.05457173E-34 # m^{2} kg s^{-1}
temperature = 290 # K
simulationTime = 5e-6


class chargeCarrier:
    def __init__(self, initialChromophore, singlesData, TIDict, boxSize, temperature):
        self.TIDict = TIDict
        self.singlesData = singlesData
        # SinglesData Dictionary is of the form:
        # key = realChromoID, val = [xPos, yPos, zPos, HOMO-1, HOMO, LUMO, LUMO+1]
        self.initialPosition = np.array(singlesData[initialChromophore][0:3])
        self.currentChromophore = initialChromophore
        self.imagePosition = [0, 0, 0]
        self.globalTime = 0
        self.T = temperature
        self.boxSize = boxSize
        self.simDims = [[-boxSize[0]/2.0, boxSize[0]/2.0], [-boxSize[1]/2.0, boxSize[1]/2.0], [-boxSize[2]/2.0, boxSize[2]/2.0]]
        self.reinitialise()

        
    def reinitialise(self):
        self.position = np.array(self.singlesData[self.currentChromophore][0:3])
        self.chromoLength = int(self.singlesData[self.currentChromophore][7])
        if plottingSubroutines == True:
            plotCarrier(self.singlesData, self.TIDict, self.currentChromophore, self.position, self.imagePosition, self.initialPosition, self.simDims, self.globalTime)
        

    def calculateLambdaij(self):
        # The equation for the internal reorganisation energy was obtained from the data given in
        # Johansson, E and Larsson, S; 2004, Synthetic Metals 144: 183-191.
        # External reorganisation energy obtained from 
        # Liu, T and Cheung, D. L. and Troisi, A; 2011, Phys. Chem. Chem. Phys. 13: 21461-21470
        lambdaExternal = 0.11 # eV
        if self.chromoLength < 12:
            lambdaInternal = 0.20826 - (self.chromoLength*0.01196)
        else:
            lambdaInternal = 0.06474
        lambdaeV = lambdaExternal+lambdaInternal
        lambdaJ = lambdaeV*elementaryCharge
        return lambdaJ

    
    def calculateEij(self, destination):
        Ei = self.singlesData[self.currentChromophore][4] # HOMO LEVEL
        Ej = self.singlesData[destination][4]
        deltaEijeV = Ej - Ei
        deltaEijJ = deltaEijeV*elementaryCharge
        return deltaEijJ

    
    def calculateHopRate(self, lambdaij, Tij, deltaEij):
        # Error in Lan 2008, should be just 1/hbar not squared
        kij = ((2*np.pi)/hbar)*(Tij**2)*np.sqrt(1.0/(4*lambdaij*np.pi*kB*self.T))*np.exp(-((deltaEij+lambdaij)**2)/(4*lambdaij*kB*self.T))
        #print "Prefactor =", ((2*np.pi)/hbar)*(Tij**2)*np.sqrt(1.0/(4*lambdaij*np.pi*kB*self.T))
        #print "Exponent =", (np.exp(-((deltaEij+lambdaij)**2)/(4*lambdaij*kB*self.T)))
        # Durham code had a different prefactor == Tij**2/hbar * sqrt(pi/(lambda*kB*T))
        return kij

    
    def determineHopTime(self, rate):
        if rate != 0:
            while True:
                x = R.random()
                if (x != 0.0) and (x != 1.0):
                    break
            tau = -np.log(x)/rate
        else:
            # Zero rate, therefore set the hop time to very long
            tau = 1E20
        return tau
    

    def calculateHop(self):
        hopTimes = []
        lambdaij = self.calculateLambdaij()
        for hopTarget in self.TIDict[self.currentChromophore]:
            #print "\n"
            transferIntegral = hopTarget[1]*elementaryCharge
            deltaEij = self.calculateEij(hopTarget[0])
            hopRate = self.calculateHopRate(lambdaij, transferIntegral, deltaEij)
            hopTime = self.determineHopTime(hopRate)
            #print "For this hop target:", hopTarget
            #print "TI =", transferIntegral
            #print "deltaEij =", deltaEij
            #print "hopRate =", hopRate
            #print "hopTime =", hopTime
            hopTimes.append([hopTarget[0], hopTime])
        hopTimes.sort(key = lambda x:x[1]) # Sort by ascending hop time
        deltaEij = self.calculateEij(hopTimes[0][0])
        #print "\n"
        #print hopTimes
        # if deltaEij <= 0.0:
        #     print "Downstream hop", deltaEij
        # else:
        #     print "Upstream hop", deltaEij
        # print "HopTime =", hopTimes[0][1]
        # raw_input('Post Hop Pause')
        self.performHop(hopTimes[0][0], hopTimes[0][1])
        return self.globalTime

        
    def performHop(self, destinationChromophore, hopTime):
        initialPosition = np.array(self.singlesData[self.currentChromophore][0:3])
        destinationPosition = np.array(self.singlesData[destinationChromophore][0:3])
        deltaPosition = destinationPosition - initialPosition
        # print "Hopping from", self.currentChromophore, "to", destinationChromophore
        # print "Current =", initialPosition, "Destination =", destinationPosition
        # print "deltaPosition =", deltaPosition
        # Work out if we've crossed a periodic boundary
        for axis in range(3):
            if np.absolute(deltaPosition[axis]) > self.boxSize[axis]/2.0:
                # Crossed a periodic boundary condition. Find out which one!
                if destinationPosition[axis] > initialPosition[axis]:
                    # Crossed a negative boundary, so increment image
                    self.imagePosition[axis] -= 1
                elif destinationPosition[axis] < initialPosition[axis]:
                    # Crossed a positive boundary
                    self.imagePosition[axis] += 1
        # Image sorted, now move the charge
        self.currentChromophore = destinationChromophore
        # Increment the time
        self.globalTime += hopTime
        self.reinitialise()


def plotCarrier(singleChromos, TIDict, currentChromo, currentPosn, image, initialPosition, simDims, globalTime):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    # Draw the simulation box first
    ax.plot([simDims[0][0], simDims[0][1]], [simDims[1][0], simDims[1][0]], [simDims[2][0], simDims[2][0]], c = 'k')
    ax.plot([simDims[0][0], simDims[0][1]], [simDims[1][1], simDims[1][1]], [simDims[2][0], simDims[2][0]], c = 'k')
    ax.plot([simDims[0][0], simDims[0][1]], [simDims[1][0], simDims[1][0]], [simDims[2][1], simDims[2][1]], c = 'k')
    ax.plot([simDims[0][0], simDims[0][1]], [simDims[1][1], simDims[1][1]], [simDims[2][1], simDims[2][1]], c = 'k')

    ax.plot([simDims[0][0], simDims[0][0]], [simDims[1][0], simDims[1][1]], [simDims[2][0], simDims[2][0]], c = 'k')
    ax.plot([simDims[0][0], simDims[0][0]], [simDims[1][0], simDims[1][1]], [simDims[2][1], simDims[2][1]], c = 'k')
    ax.plot([simDims[0][1], simDims[0][1]], [simDims[1][0], simDims[1][1]], [simDims[2][0], simDims[2][0]], c = 'k')
    ax.plot([simDims[0][1], simDims[0][1]], [simDims[1][0], simDims[1][1]], [simDims[2][1], simDims[2][1]], c = 'k')

    ax.plot([simDims[0][0], simDims[0][0]], [simDims[1][0], simDims[1][0]], [simDims[2][0], simDims[2][1]], c = 'k')
    ax.plot([simDims[0][0], simDims[0][0]], [simDims[1][1], simDims[1][1]], [simDims[2][0], simDims[2][1]], c = 'k')
    ax.plot([simDims[0][1], simDims[0][1]], [simDims[1][0], simDims[1][0]], [simDims[2][0], simDims[2][1]], c = 'k')
    ax.plot([simDims[0][1], simDims[0][1]], [simDims[1][1], simDims[1][1]], [simDims[2][0], simDims[2][1]], c = 'k')

    # Draw current chromo position
    ax.scatter(currentPosn[0], currentPosn[1], currentPosn[2], s = 50, c = 'r')
    # Draw neighbouring chromos
    for hopOption in TIDict[currentChromo]:
        hopOptionPosn = np.array(singleChromos[hopOption[0]][0:3])
        ax.scatter(hopOptionPosn[0], hopOptionPosn[1], hopOptionPosn[2], s = 20, c = 'b')

    # Complete plot
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    extraSpace = 5
    ax.set_xlim(simDims[0][0]-extraSpace, simDims[0][1]+extraSpace)
    ax.set_ylim(simDims[1][0]-extraSpace, simDims[1][1]+extraSpace)
    ax.set_zlim(simDims[2][0]-extraSpace, simDims[2][1]+extraSpace)

    displacement = helperFunctions.calculateSeparation(initialPosition, currentPosn)
    displacement = list(str(displacement))
    displacementString = ''
    for i in range(5):
        try:
            displacementString += displacement[i]
        except IndexError:
            break

    plt.title(str(image)+", Disp ="+str(displacementString)+", t = "+str(globalTime))
    fileList = os.listdir('./')
    fileNameCounter = 0
    for files in fileList:
        if ".png" in files:
            fileNameCounter += 1
    fileNameAddon = str(fileNameCounter)
    while len(fileNameAddon) < 3:
        fileNameAddon = '0'+fileNameAddon
    plt.savefig('./test'+fileNameAddon+'.png')
    # plt.show()
    # raw_input('Break for Ctrl-C')
    print "Image saved as ./test"+fileNameAddon+".png"
    plt.close(fig)


def randomPosition(boxSize, singleChromos):
    randX = R.uniform(-boxSize[0]/2.0, boxSize[0]/2.0)
    randY = R.uniform(-boxSize[1]/2.0, boxSize[1]/2.0) 
    randZ = R.uniform(-boxSize[2]/2.0, boxSize[2]/2.0)
    randomPosn = [randX, randY, randZ]

    separationToSegments = [] # Of the form [ [seg1No, seg1Sep], [seg2No, seg2Sep] ... ]
    for segNo, chromo in singleChromos.iteritems():
        separationToSegments.append([segNo, helperFunctions.calculateSeparation(randomPosn, np.array(chromo[1:4]))])
    separationToSegments.sort(key = lambda x: x[1])
    return int(separationToSegments[0][0])
    
    
def execute(morphologyName, boxSize):
    R.seed(32)
    CSVDir = os.getcwd()+'/outputFiles/'+morphologyName+'/chromophores'
    singlesData = {}
    pairsData = []
    try:
        with open(CSVDir+'/singles.csv', 'r') as singlesFile:
            singlesReader = csv.reader(singlesFile, delimiter=',')
            for row in singlesReader:
                singlesData[int(float(row[0]))] = map(float, row[1:])
        with open(CSVDir+'/pairs.csv', 'r') as pairsFile:
            pairsReader = csv.reader(pairsFile, delimiter=',')
            for row in pairsReader:
                pairsData.append([int(float(row[0])), int(float(row[1]))] + [x for x in map(float, row[2:])])
    except IOError:
        print "CSV files singles.csv and pairs.csv not found in the chromophores directory."
        print "Please run transferIntegrals.py to generate these files from the ORCA outputs."
        return
    TIDict = {}
    totalPairs = 0
    numberOfZeroes = 0
    for pair in pairsData:
        if pair[0] not in TIDict:
            TIDict[pair[0]] = []
        if pair[1] not in TIDict:
            TIDict[pair[1]] = []
        TIDict[pair[0]].append([pair[1], pair[-1]]) # Neighbour and corresponding Tij
        TIDict[pair[1]].append([pair[0], pair[-1]]) # Reverse Hop
        if pair[-1] == 0.0:
            numberOfZeroes += 1
        totalPairs += 1

    print "There are", totalPairs, "total possible hop destinations, and", numberOfZeroes, "of them have a transfer integral of zero ("+str(int(float(numberOfZeroes)/float(totalPairs)*100))+"%)..."
    # Loop Start Here
    # Pick a random chromophore to inject to
    initialChromophore = randomPosition(boxSize, singlesData)
    #print "Injecting onto", initialChromophore, "TIDict =", TIDict[initialChromophore]
    # Initialise a carrier
    hole = chargeCarrier(initialChromophore, singlesData, TIDict, boxSize, temperature)
    numberOfHops = 0
    newGlobalTime = 0.0
    # Start hopping!
    while True:
        if plottingSubroutines == True:
            if numberOfHops == 100:
                break
            print "Performing hop number", numberOfHops+1
        else:
            if newGlobalTime > simulationTime:
                break
        hole.calculateHop()
        numberOfHops += 1

    initialPos = hole.initialPosition
    currentPos = np.array([hole.position[0]+(hole.imagePosition[0]*boxSize[0]), hole.position[1]+(hole.imagePosition[1]*boxSize[1]), hole.position[2]+(hole.imagePosition[2]*boxSize[2])])
    displacement = helperFunctions.calculateSeparation(hole.initialPosition, hole.position)
    # Update CSV file
    if plottingSubroutines == False:
        csvFileName = "./CTOutput/"+morphology+"_"+str(simulationTime)+"_Koop"+str(koopmansApproximation)[0]+".csv"
        print numberOfHops, "hops complete. Writing displacement of", displacement, "for carrier number", carrierNo, " in", csvFileName
        writeCSVFile(csvFileName, carrierNo, displacement)
    else:
        print numberOfHops, "hops complete. Graphs plotted. Simulation terminating. No CSV data will be saved while plotting == True."

    
    
def loadPickle(morphologyFile):
    morphologyName = morphologyFile[helperFunctions.findIndex(morphologyFile,'/')[-1]+1:]
    outputDir = './outputFiles'
    morphologyList = os.listdir(outputDir)
    for allMorphologies in morphologyList:
        if morphologyName in allMorphologies:
            outputDir += '/'+morphologyName
            break
    pickleFound = False
    for fileName in os.listdir(outputDir+'/morphology'):
        if fileName == morphologyName+'.pickle':
            pickleLoc = outputDir+'/morphology/'+fileName
            pickleFound = True
    if pickleFound == False:
        print "Pickle file not found. Please run morphCT.py again to create the required HOOMD inputs."
        exit()
    print "Pickle found at", str(pickleLoc)+"."
    print "Loading data..."
    with open(pickleLoc, 'r') as pickleFile:
        (AAfileName, CGMoleculeDict, AAMorphologyDict, CGtoAAIDs, moleculeAAIDs, boxSize) = pickle.load(pickleFile)
    execute(morphologyName, boxSize)
    return morphologyFile, AAfileName, CGMoleculeDict, AAMorphologyDict, CGtoAAIDs, moleculeAAIDs, boxSize


if __name__ == "__main__":
    morphologyFile = sys.argv[1]
    loadPickle(morphologyFile)
