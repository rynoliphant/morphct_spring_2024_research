import os
import numpy as np
import time as T
import helperFunctions

def countOutputFiles(directory):
    singleOutputs = os.listdir(directory+'/single/')
    pairOutputs = os.listdir(directory+'/pair/')
    orcaOutputs = 0
    for fileName in singleOutputs:
        if fileName[-4:] == '.out':
            orcaOutputs += 1
    for fileName in pairOutputs:
        if fileName[-4:] == '.out':
            orcaOutputs += 1
    return orcaOutputs


def execute(morphologyFile, slurmJobNumber):
    morphologyName = morphologyFile[helperFunctions.findIndex(morphologyFile,'/')[-1]+1:]
    inputDir = os.getcwd()+'/outputFiles/'+morphologyName+'/chromophores/inputORCA'
    # Clear input files
    try:
        os.unlink(inputDir.replace('/inputORCA', '/*.log'))
    except OSError:
        pass
    procIDs, jobsList = helperFunctions.getORCAJobs(inputDir)
    numberOfInputs = sum([len(ORCAFilesToRun) for ORCAFilesToRun in jobsList])
    print "Found", numberOfInputs, "ORCA files to run."
    print procIDs
    for CPURank in procIDs:
        print 'python '+os.getcwd()+'/code/singleCoreRunORCA.py '+os.getcwd()+'/outputFiles/'+morphologyName+' '+str(CPURank)+' &'
        os.system('python '+os.getcwd()+'/code/singleCoreRunORCA.py '+os.getcwd()+'/outputFiles/'+morphologyName+' '+str(CPURank)+' &')

    print "Checking for completed output files..."
    previousNumberOfOutputs = -1
    slurmCancel = False
    while True:
        if slurmCancel == True:
            print "Terminating program..."
            os.system('scancel '+slurmJobNumber)
            exit()
        numberOfOutputs = countOutputFiles(os.getcwd()+'/outputFiles/'+morphologyName+'/chromophores/outputORCA')
        if numberOfOutputs == numberOfInputs:
            print "All", numberOfInputs, "output files present. Waiting one more iteration for current jobs to complete..."
            slurmCancel = True
        if numberOfOutputs == previousNumberOfOutputs:
            print "No additional output files found this iteration - there are still", numberOfOutputs, "output files present. Is everything still working?"
        previousNumberOfOutputs = numberOfOutputs
        # Sleep for 20 minutes
        T.sleep(1200)

if __name__ == '__main__':
    morphologyFile = sys.argv[1]
    slurmJobNumber = sys.argv[2]
    execute(morphologyFile, slurmJobNumber)
