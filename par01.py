import runMorphCT

# ---=== Directory and File Structure ===---
inputDir = '/Users/mattyjones/GoogleDrive/Boise/Code/MorphCT/inputCGMorphs'
outputDir = '/Users/mattyjones/GoogleDrive/Boise/Code/MorphCT/outputFiles'


# ---=== Input Morphology Details ===---
morphology = 'p1-L15-f0.0-P0.1-T1.5-e0.5.xml'
inputSigma = 3.0
overwriteCurrentData = True

# ---=== Execution Modules ===---

executeFinegraining = True
executeMolecularDynamics = True
executeExtractMolecules = True
executeObtainChromophores = True
executeZINDOS = True
executeCalculateTransferIntegrals = True
executeCalculateMobility = True

# ---=== Fine Graining Parameters ===---

repeatUnitTemplateDirectory = '/Users/mattyjones/GoogleDrive/Boise/Code/MorphCT/templates'
repeatUnitTemplateFile = 'mid3HT.xml'
CGToTemplateAAIDs = {\
'A':[0, 1, 2, 3, 4, 24],\
'B':[5, 6, 7, 18, 19, 20, 21, 22, 23],\
'C':[8, 9, 10, 11, 12, 13, 14, 15, 16, 17],\
}
CGToTemplateBonds = {\
'bondA':['C1-C10', 3, 25],\
'bondB':['C2-C3', 2, 5],\
'bondC':['C5-C6', 7, 8],\
}
### NEED TO INCLUDE RIGID BODIES HERE ###

# ---=== Forcefield Parameters ===---
pairRCut = 10
pairDPDGammaVal = 0.0
# --== Lennard-Jones Pair ==--
ljCoeffs = [\
['C1-C1', 1.0, 1.0],\
['C1-C10', 1.0, 1.0],\
['C1-C2', 1.0, 1.0],\
['C1-C3', 1.0, 1.0],\
['C1-C4', 1.0, 1.0],\
['C1-C5', 1.0, 1.0],\
['C1-C6', 1.0, 1.0],\
['C1-C7', 1.0, 1.0],\
['C1-C8', 1.0, 1.0],\
['C1-C9', 1.0, 1.0],\
['C1-H1', 1.0, 1.0],\
['C1-S1', 1.0, 1.0],\
['C10-C1', 1.0, 1.0],\
['C10-C10', 1.0, 1.0],\
['C10-C2', 1.0, 1.0],\
['C10-C3', 1.0, 1.0],\
['C10-C4', 1.0, 1.0],\
['C10-C5', 1.0, 1.0],\
['C10-C6', 1.0, 1.0],\
['C10-C7', 1.0, 1.0],\
['C10-C8', 1.0, 1.0],\
['C10-C9', 1.0, 1.0],\
['C10-H1', 1.0, 1.0],\
['C10-S1', 1.0, 1.0],\
['C2-C1', 1.0, 1.0],\
['C2-C10', 1.0, 1.0],\
['C2-C2', 1.0, 1.0],\
['C2-C3', 1.0, 1.0],\
['C2-C4', 1.0, 1.0],\
['C2-C5', 1.0, 1.0],\
['C2-C6', 1.0, 1.0],\
['C2-C7', 1.0, 1.0],\
['C2-C8', 1.0, 1.0],\
['C2-C9', 1.0, 1.0],\
['C2-H1', 1.0, 1.0],\
['C2-S1', 1.0, 1.0],\
['C3-C1', 1.0, 1.0],\
['C3-C10', 1.0, 1.0],\
['C3-C2', 1.0, 1.0],\
['C3-C3', 1.0, 1.0],\
['C3-C4', 1.0, 1.0],\
['C3-C5', 1.0, 1.0],\
['C3-C6', 1.0, 1.0],\
['C3-C7', 1.0, 1.0],\
['C3-C8', 1.0, 1.0],\
['C3-C9', 1.0, 1.0],\
['C3-H1', 1.0, 1.0],\
['C3-S1', 1.0, 1.0],\
['C4-C1', 1.0, 1.0],\
['C4-C10', 1.0, 1.0],\
['C4-C2', 1.0, 1.0],\
['C4-C3', 1.0, 1.0],\
['C4-C4', 1.0, 1.0],\
['C4-C5', 1.0, 1.0],\
['C4-C6', 1.0, 1.0],\
['C4-C7', 1.0, 1.0],\
['C4-C8', 1.0, 1.0],\
['C4-C9', 1.0, 1.0],\
['C4-H1', 1.0, 1.0],\
['C4-S1', 1.0, 1.0],\
['C5-C1', 1.0, 1.0],\
['C5-C10', 1.0, 1.0],\
['C5-C2', 1.0, 1.0],\
['C5-C3', 1.0, 1.0],\
['C5-C4', 1.0, 1.0],\
['C5-C5', 1.0, 1.0],\
['C5-C6', 1.0, 1.0],\
['C5-C7', 1.0, 1.0],\
['C5-C8', 1.0, 1.0],\
['C5-C9', 1.0, 1.0],\
['C5-H1', 1.0, 1.0],\
['C5-S1', 1.0, 1.0],\
['C6-C1', 1.0, 1.0],\
['C6-C10', 1.0, 1.0],\
['C6-C2', 1.0, 1.0],\
['C6-C3', 1.0, 1.0],\
['C6-C4', 1.0, 1.0],\
['C6-C5', 1.0, 1.0],\
['C6-C6', 1.0, 1.0],\
['C6-C7', 1.0, 1.0],\
['C6-C8', 1.0, 1.0],\
['C6-C9', 1.0, 1.0],\
['C6-H1', 1.0, 1.0],\
['C6-S1', 1.0, 1.0],\
['C7-C1', 1.0, 1.0],\
['C7-C10', 1.0, 1.0],\
['C7-C2', 1.0, 1.0],\
['C7-C3', 1.0, 1.0],\
['C7-C4', 1.0, 1.0],\
['C7-C5', 1.0, 1.0],\
['C7-C6', 1.0, 1.0],\
['C7-C7', 1.0, 1.0],\
['C7-C8', 1.0, 1.0],\
['C7-C9', 1.0, 1.0],\
['C7-H1', 1.0, 1.0],\
['C7-S1', 1.0, 1.0],\
['C8-C1', 1.0, 1.0],\
['C8-C10', 1.0, 1.0],\
['C8-C2', 1.0, 1.0],\
['C8-C3', 1.0, 1.0],\
['C8-C4', 1.0, 1.0],\
['C8-C5', 1.0, 1.0],\
['C8-C6', 1.0, 1.0],\
['C8-C7', 1.0, 1.0],\
['C8-C8', 1.0, 1.0],\
['C8-C9', 1.0, 1.0],\
['C8-H1', 1.0, 1.0],\
['C8-S1', 1.0, 1.0],\
['C9-C1', 1.0, 1.0],\
['C9-C10', 1.0, 1.0],\
['C9-C2', 1.0, 1.0],\
['C9-C3', 1.0, 1.0],\
['C9-C4', 1.0, 1.0],\
['C9-C5', 1.0, 1.0],\
['C9-C6', 1.0, 1.0],\
['C9-C7', 1.0, 1.0],\
['C9-C8', 1.0, 1.0],\
['C9-C9', 1.0, 1.0],\
['C9-H1', 1.0, 1.0],\
['C9-S1', 1.0, 1.0],\
['H1-C1', 1.0, 1.0],\
['H1-C10', 1.0, 1.0],\
['H1-C2', 1.0, 1.0],\
['H1-C3', 1.0, 1.0],\
['H1-C4', 1.0, 1.0],\
['H1-C5', 1.0, 1.0],\
['H1-C6', 1.0, 1.0],\
['H1-C7', 1.0, 1.0],\
['H1-C8', 1.0, 1.0],\
['H1-C9', 1.0, 1.0],\
['H1-H1', 1.0, 1.0],\
['H1-S1', 1.0, 1.0],\
['S1-C1', 1.0, 1.0],\
['S1-C10', 1.0, 1.0],\
['S1-C2', 1.0, 1.0],\
['S1-C3', 1.0, 1.0],\
['S1-C4', 1.0, 1.0],\
['S1-C5', 1.0, 1.0],\
['S1-C6', 1.0, 1.0],\
['S1-C7', 1.0, 1.0],\
['S1-C8', 1.0, 1.0],\
['S1-C9', 1.0, 1.0],\
['S1-H1', 1.0, 1.0],\
['S1-S1', 1.0, 1.0],\
]
# --== Disipative Particle Dynamics Pair ==--
dpdCoeffs = [\
['C1-C1', 1.0, 1.0],\
['C1-C10', 1.0, 1.0],\
['C1-C2', 1.0, 1.0],\
['C1-C3', 1.0, 1.0],\
['C1-C4', 1.0, 1.0],\
['C1-C5', 1.0, 1.0],\
['C1-C6', 1.0, 1.0],\
['C1-C7', 1.0, 1.0],\
['C1-C8', 1.0, 1.0],\
['C1-C9', 1.0, 1.0],\
['C1-H1', 1.0, 1.0],\
['C1-S1', 1.0, 1.0],\
['C10-C1', 1.0, 1.0],\
['C10-C10', 1.0, 1.0],\
['C10-C2', 1.0, 1.0],\
['C10-C3', 1.0, 1.0],\
['C10-C4', 1.0, 1.0],\
['C10-C5', 1.0, 1.0],\
['C10-C6', 1.0, 1.0],\
['C10-C7', 1.0, 1.0],\
['C10-C8', 1.0, 1.0],\
['C10-C9', 1.0, 1.0],\
['C10-H1', 1.0, 1.0],\
['C10-S1', 1.0, 1.0],\
['C2-C1', 1.0, 1.0],\
['C2-C10', 1.0, 1.0],\
['C2-C2', 1.0, 1.0],\
['C2-C3', 1.0, 1.0],\
['C2-C4', 1.0, 1.0],\
['C2-C5', 1.0, 1.0],\
['C2-C6', 1.0, 1.0],\
['C2-C7', 1.0, 1.0],\
['C2-C8', 1.0, 1.0],\
['C2-C9', 1.0, 1.0],\
['C2-H1', 1.0, 1.0],\
['C2-S1', 1.0, 1.0],\
['C3-C1', 1.0, 1.0],\
['C3-C10', 1.0, 1.0],\
['C3-C2', 1.0, 1.0],\
['C3-C3', 1.0, 1.0],\
['C3-C4', 1.0, 1.0],\
['C3-C5', 1.0, 1.0],\
['C3-C6', 1.0, 1.0],\
['C3-C7', 1.0, 1.0],\
['C3-C8', 1.0, 1.0],\
['C3-C9', 1.0, 1.0],\
['C3-H1', 1.0, 1.0],\
['C3-S1', 1.0, 1.0],\
['C4-C1', 1.0, 1.0],\
['C4-C10', 1.0, 1.0],\
['C4-C2', 1.0, 1.0],\
['C4-C3', 1.0, 1.0],\
['C4-C4', 1.0, 1.0],\
['C4-C5', 1.0, 1.0],\
['C4-C6', 1.0, 1.0],\
['C4-C7', 1.0, 1.0],\
['C4-C8', 1.0, 1.0],\
['C4-C9', 1.0, 1.0],\
['C4-H1', 1.0, 1.0],\
['C4-S1', 1.0, 1.0],\
['C5-C1', 1.0, 1.0],\
['C5-C10', 1.0, 1.0],\
['C5-C2', 1.0, 1.0],\
['C5-C3', 1.0, 1.0],\
['C5-C4', 1.0, 1.0],\
['C5-C5', 1.0, 1.0],\
['C5-C6', 1.0, 1.0],\
['C5-C7', 1.0, 1.0],\
['C5-C8', 1.0, 1.0],\
['C5-C9', 1.0, 1.0],\
['C5-H1', 1.0, 1.0],\
['C5-S1', 1.0, 1.0],\
['C6-C1', 1.0, 1.0],\
['C6-C10', 1.0, 1.0],\
['C6-C2', 1.0, 1.0],\
['C6-C3', 1.0, 1.0],\
['C6-C4', 1.0, 1.0],\
['C6-C5', 1.0, 1.0],\
['C6-C6', 1.0, 1.0],\
['C6-C7', 1.0, 1.0],\
['C6-C8', 1.0, 1.0],\
['C6-C9', 1.0, 1.0],\
['C6-H1', 1.0, 1.0],\
['C6-S1', 1.0, 1.0],\
['C7-C1', 1.0, 1.0],\
['C7-C10', 1.0, 1.0],\
['C7-C2', 1.0, 1.0],\
['C7-C3', 1.0, 1.0],\
['C7-C4', 1.0, 1.0],\
['C7-C5', 1.0, 1.0],\
['C7-C6', 1.0, 1.0],\
['C7-C7', 1.0, 1.0],\
['C7-C8', 1.0, 1.0],\
['C7-C9', 1.0, 1.0],\
['C7-H1', 1.0, 1.0],\
['C7-S1', 1.0, 1.0],\
['C8-C1', 1.0, 1.0],\
['C8-C10', 1.0, 1.0],\
['C8-C2', 1.0, 1.0],\
['C8-C3', 1.0, 1.0],\
['C8-C4', 1.0, 1.0],\
['C8-C5', 1.0, 1.0],\
['C8-C6', 1.0, 1.0],\
['C8-C7', 1.0, 1.0],\
['C8-C8', 1.0, 1.0],\
['C8-C9', 1.0, 1.0],\
['C8-H1', 1.0, 1.0],\
['C8-S1', 1.0, 1.0],\
['C9-C1', 1.0, 1.0],\
['C9-C10', 1.0, 1.0],\
['C9-C2', 1.0, 1.0],\
['C9-C3', 1.0, 1.0],\
['C9-C4', 1.0, 1.0],\
['C9-C5', 1.0, 1.0],\
['C9-C6', 1.0, 1.0],\
['C9-C7', 1.0, 1.0],\
['C9-C8', 1.0, 1.0],\
['C9-C9', 1.0, 1.0],\
['C9-H1', 1.0, 1.0],\
['C9-S1', 1.0, 1.0],\
['H1-C1', 1.0, 1.0],\
['H1-C10', 1.0, 1.0],\
['H1-C2', 1.0, 1.0],\
['H1-C3', 1.0, 1.0],\
['H1-C4', 1.0, 1.0],\
['H1-C5', 1.0, 1.0],\
['H1-C6', 1.0, 1.0],\
['H1-C7', 1.0, 1.0],\
['H1-C8', 1.0, 1.0],\
['H1-C9', 1.0, 1.0],\
['H1-H1', 1.0, 1.0],\
['H1-S1', 1.0, 1.0],\
['S1-C1', 1.0, 1.0],\
['S1-C10', 1.0, 1.0],\
['S1-C2', 1.0, 1.0],\
['S1-C3', 1.0, 1.0],\
['S1-C4', 1.0, 1.0],\
['S1-C5', 1.0, 1.0],\
['S1-C6', 1.0, 1.0],\
['S1-C7', 1.0, 1.0],\
['S1-C8', 1.0, 1.0],\
['S1-C9', 1.0, 1.0],\
['S1-H1', 1.0, 1.0],\
['S1-S1', 1.0, 1.0],\
]
# --== Bond ==--
bondCoeffs = [\
['C1-C2', 1.0, 1.0],\
['C1-S1', 1.0, 1.0],\
['C10-C9', 1.0, 1.0],\
['C10-S1', 1.0, 1.0],\
['C2-C3', 1.0, 1.0],\
['C2-C9', 1.0, 1.0],\
['C3-C4', 1.0, 1.0],\
['C3-H1', 1.0, 1.0],\
['C4-C5', 1.0, 1.0],\
['C4-H1', 1.0, 1.0],\
['C5-C6', 1.0, 1.0],\
['C5-H1', 1.0, 1.0],\
['C6-C7', 1.0, 1.0],\
['C6-H1', 1.0, 1.0],\
['C7-C8', 1.0, 1.0],\
['C7-H1', 1.0, 1.0],\
['C8-H1', 1.0, 1.0],\
['C9-H1', 1.0, 1.0],\
]
# --== Angle ==--
angleCoeffs = [\
['C1-C10', 1.0, 1.0],\
['C1-C2-C3', 1.0, 1.0],\
['C1-C2-C9', 1.0, 1.0],\
['C10-C9-C2', 1.0, 1.0],\
['C10-C9-H1', 1.0, 1.0],\
['C10-S1-C1', 1.0, 1.0],\
['C2-C1-S1', 1.0, 1.0],\
['C2-C3-C4', 1.0, 1.0],\
['C2-C3-H1', 1.0, 1.0],\
['C2-C9-H1', 1.0, 1.0],\
['C3-C2-C9', 1.0, 1.0],\
['C3-C4-C5', 1.0, 1.0],\
['C3-C4-H1', 1.0, 1.0],\
['C4-C3-H1', 1.0, 1.0],\
['C4-C5-C6', 1.0, 1.0],\
['C4-C5-H1', 1.0, 1.0],\
['C5-C4-H1', 1.0, 1.0],\
['C5-C6-C7', 1.0, 1.0],\
['C5-C6-H1', 1.0, 1.0],\
['C6-C5-H1', 1.0, 1.0],\
['C6-C7-C8', 1.0, 1.0],\
['C6-C7-H1', 1.0, 1.0],\
['C7-C6-H1', 1.0, 1.0],\
['C7-C8-H1', 1.0, 1.0],\
['C8-C7-H1', 1.0, 1.0],\
['C9-C10-S1', 1.0, 1.0],\
['H1-C3-H1', 1.0, 1.0],\
['H1-C4-H1', 1.0, 1.0],\
['H1-C5-H1', 1.0, 1.0],\
['H1-C6-H1', 1.0, 1.0],\
['H1-C7-H1', 1.0, 1.0],\
['H1-C8-H1', 1.0, 1.0],\
]
# --== Dihedral ==--
dihedralCoeffs = [\
['C1-C10-C9', 1.0, 1.0, 1.0, 1.0],\
['C1-C10-S1', 1.0, 1.0, 1.0, 1.0],\
['C1-C2-C3-C4', 1.0, 1.0, 1.0, 1.0],\
['C1-C2-C3-H1', 1.0, 1.0, 1.0, 1.0],\
['C1-C2-C9-H1', 1.0, 1.0, 1.0, 1.0],\
['C1-S1-C10-C9', 1.0, 1.0, 1.0, 1.0],\
['C10-C2-C9-C1', 1.0, 1.0, 1.0, 1.0],\
['C10-C9-C2-C3', 1.0, 1.0, 1.0, 1.0],\
['C10-S1-C1-C2', 1.0, 1.0, 1.0, 1.0],\
['C2-C1-C10', 1.0, 1.0, 1.0, 1.0],\
['C2-C3-C4-C5', 1.0, 1.0, 1.0, 1.0],\
['C2-C3-C4-H1', 1.0, 1.0, 1.0, 1.0],\
['C2-C9-C10-S1', 1.0, 1.0, 1.0, 1.0],\
['C3-C2-C1-S1', 1.0, 1.0, 1.0, 1.0],\
['C3-C2-C9-H1', 1.0, 1.0, 1.0, 1.0],\
['C3-C4-C5-C6', 1.0, 1.0, 1.0, 1.0],\
['C3-C4-C5-H1', 1.0, 1.0, 1.0, 1.0],\
['C4-C3-C2-C9', 1.0, 1.0, 1.0, 1.0],\
['C4-C5-C6-C7', 1.0, 1.0, 1.0, 1.0],\
['C4-C5-C6-H1', 1.0, 1.0, 1.0, 1.0],\
['C5-C4-C3-H1', 1.0, 1.0, 1.0, 1.0],\
['C5-C6-C7-C8', 1.0, 1.0, 1.0, 1.0],\
['C5-C6-C7-H1', 1.0, 1.0, 1.0, 1.0],\
['C6-C5-C4-H1', 1.0, 1.0, 1.0, 1.0],\
['C6-C7-C8-H1', 1.0, 1.0, 1.0, 1.0],\
['C7-C6-C5-H1', 1.0, 1.0, 1.0, 1.0],\
['C8-C7-C6-H1', 1.0, 1.0, 1.0, 1.0],\
['C9-C2-C1-S1', 1.0, 1.0, 1.0, 1.0],\
['C9-C2-C3-H1', 1.0, 1.0, 1.0, 1.0],\
['H1-C3-C4-H1', 1.0, 1.0, 1.0, 1.0],\
['H1-C4-C5-H1', 1.0, 1.0, 1.0, 1.0],\
['H1-C5-C6-H1', 1.0, 1.0, 1.0, 1.0],\
['H1-C6-C7-H1', 1.0, 1.0, 1.0, 1.0],\
['H1-C7-C8-H1', 1.0, 1.0, 1.0, 1.0],\
['H1-C9-C10-S1', 1.0, 1.0, 1.0, 1.0],\
['S1-C1-C10', 1.0, 1.0, 1.0, 1.0],\
]
# --== Improper ==--
improperCoeffs = [\
['C1-C10-C9-C2', 1.0, 1.0, 1.0, 1.0],\
['C1-C10-S1-C1', 1.0, 1.0, 1.0, 1.0],\
['C2-C1-C10-S1', 1.0, 1.0, 1.0, 1.0],\
['S1-C1-C10-S1', 1.0, 1.0, 1.0, 1.0],\
]

# ---=== Molecular Dynamics Phase Parameters ===---
numberOfPhases = 6
temperatures = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
pairType = ['none', 'dpd', 'lj', 'lj', 'lj', 'lj']
bondType = 'harmonic'
angleType = 'harmonic'
dihedralType = 'table'
integrationTargets = ['all', 'sidechains', 'all', 'all', 'all', 'all']
timesteps = [1E-3, 1E-3, 1E-9, 1E-7, 1E-6, 1E-5]
phaseDurations = [1E3, 1E4, 1E2, 1E2, 1E5, 1E5]
terminationConditions = ['KEmin', 'maxt', 'maxt', 'maxt', 'maxt', 'maxt']
groupAnchoring = ['all', 'all', 'all', 'all', 'all', 'none']


# ---=== Begin run ===---
parameterFile = __file__

if __name__ == "__main__":
    parameterNames = [i for i in dir() if (not i.startswith('__')) and (i not in ['runMorphCT'])]
    parameters = {}
    for name in parameterNames:
        parameters[name] = locals()[name]
    runMorphCT.simulation(**parameters) # Execute MorphCT using these simulation parameters